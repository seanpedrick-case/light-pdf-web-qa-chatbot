import datetime
import os
import re
import time
from itertools import compress
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple, Type
from queue import Empty

# For Name Entity Recognition model
# from span_marker import SpanMarkerModel # Not currently used
# For BM25 retrieval
import bm25s
import boto3
import gradio as gr
import pandas as pd
import Stemmer

# Model packages
import torch.cuda
from google import genai as ai
from google.genai import types
from gradio import Progress
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, pipeline

from tools.config import (
    AWS_DEFAULT_REGION,
    AWS_MODELS,
    BEDROCK_MODEL_ID,
    FEEDBACK_LOGS_FOLDER,
    GEMINI_API_KEY,
    GENERATION_FIRST_TOKEN_TIMEOUT,
    GENERATION_TOKEN_TIMEOUT,
    LARGE_MODEL_NAME,
    RUN_AWS_FUNCTIONS,
    SMALL_MODEL_NAME,
    USE_BEDROCK_KB,
)
from tools.document import Document
from tools.embeddings import HuggingFaceEmbeddings
from tools.faiss_store import FAISS
from tools.model_load import (
    CtransGenGenerationConfig,
    max_new_tokens,
    max_tokens,
    repetition_penalty,
    sample,
    temperature,
    top_k,
    top_p,
    torch_device,
)
from tools.prompts import (
    instruction_prompt_gemma,
    instruction_prompt_phi3,
    instruction_prompt_qwen,
    instruction_prompt_template_gemini_aws,
)

model_object = []  # Define empty list for model functions to run
tokenizer = []  # Define empty list for model functions to run

# Shared cancel flag for local Hugging Face generate (Stop button / timeouts).
_generation_cancel = Event()


class _CancelOnEvent(StoppingCriteria):
    """Abort model.generate when Stop is clicked or a timeout invalidates the run."""

    def __call__(self, input_ids, scores, **kwargs):
        return _generation_cancel.is_set()


def begin_generation_run() -> None:
    _generation_cancel.clear()


def request_generation_cancel():
    """Cancel active local generation and re-enable the chat controls."""
    _generation_cancel.set()
    print("Generation cancel requested (Stop / timeout).")
    return restore_interactivity()


def _iter_text_streamer(
    streamer: TextIteratorStreamer,
    *,
    first_token_timeout: float,
    token_timeout: float,
    poll_interval: float = 0.5,
):
    """Yield streamer chunks with first-token / between-token deadlines.

    Polls in short intervals so Stop can abort without waiting for the full
    timeout window (a single long ``queue.get`` would ignore cancel until then).
    """
    got_first = False
    deadline = time.monotonic() + first_token_timeout
    while True:
        if _generation_cancel.is_set():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kind = "first token" if not got_first else "next token"
            limit = first_token_timeout if not got_first else token_timeout
            raise TimeoutError(
                f"Timed out waiting for {kind} after {limit:.0f}s. "
                "The previous request was cancelled — please try again."
            )
        try:
            value = streamer.text_queue.get(timeout=min(poll_interval, remaining))
        except Empty:
            continue
        if value == streamer.stop_signal:
            break
        if value:
            got_first = True
        # Reset the between-token clock after each chunk (including blanks).
        deadline = time.monotonic() + (
            token_timeout if got_first else first_token_timeout
        )
        yield value


# ResponseObject class for AWS Bedrock calls
class ResponseObject:
    def __init__(self, text, usage_metadata):
        self.text = text
        self.usage_metadata = usage_metadata


if RUN_AWS_FUNCTIONS == "1":
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_DEFAULT_REGION)
else:
    bedrock_runtime = ""

torch.cuda.empty_cache()

PandasDataFrame = Type[pd.DataFrame]

embeddings = None  # global variable setup
embeddings_model = None  # global variable setup
vectorstore = None  # global variable setup
model_type = None  # global variable setup

max_memory_length = 0  # How long should the memory of the conversation last?

source_texts = (
    ""  # Define dummy source text (full text) just to enable highlight function to load
)

## Highlight text constants
# Minimum consecutive complete words that must match the reply before highlighting
hlt_min_match_words = 5

## Initialise NER model ##
ner_model = (
    []
)  # SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd") # Not currently used

## Initialise keyword model ##
# Used to pull out keywords from chat history to add to user queries behind the scenes
kw_model = pipeline(
    "feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2"
)


def strip_model_artifacts(text: str, *, finalize: bool = False) -> str:
    """Remove Qwen-style thinking blocks and leaked role prefixes from model output."""
    if not text:
        return ""

    cleaned = str(text)
    # Complete thinking / reasoning blocks (including empty ones).
    cleaned = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<thinking\b[^>]*>.*?</thinking>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Hide an unclosed thinking block while tokens are still streaming.
    cleaned = re.sub(
        r"<think\b[^>]*>.*\Z",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<thinking\b[^>]*>.*\Z",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Orphan closing tags and common role leaks from chat templates.
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"</thinking>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^\s*(assistant|model)\s*\n+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(assistant|model)\s*[:\-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    if finalize:
        cleaned = cleaned.strip()
    return cleaned


def base_prompt_templates(model_type: str = SMALL_MODEL_NAME):

    # Simple string template for content
    CONTENT_PROMPT_TEMPLATE = "{page_content}\n\n"

    # The main prompt:
    model_l = (model_type or "").lower()
    if "qwen" in model_l:
        INSTRUCTION_PROMPT_TEMPLATE = instruction_prompt_qwen
    elif model_type == SMALL_MODEL_NAME:
        INSTRUCTION_PROMPT_TEMPLATE = instruction_prompt_gemma
    elif model_type == LARGE_MODEL_NAME:
        INSTRUCTION_PROMPT_TEMPLATE = instruction_prompt_phi3
    else:
        INSTRUCTION_PROMPT_TEMPLATE = instruction_prompt_template_gemini_aws

    return INSTRUCTION_PROMPT_TEMPLATE, CONTENT_PROMPT_TEMPLATE


# Metadata keys useful for retrieval internals but noisy in the sources panel.
_META_SKIP_KEYS = {"page_section", "end_line"}
_META_DISPLAY_LABELS = {
    "start_line": "paragraphs",
    "page": "page",
    "section": "section",
    "source": "source",
    "date": "date",
    "row": "row",
    "row_section": "row section",
}


def _metadata_value_is_empty(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    # Bad merges of empty values can leave "to" / "–" alone.
    if text in {"to", "-", "–", "to to"}:
        return True
    if re.fullmatch(r"to(\s+to)*", text):
        return True
    return False


def _parse_line_bound(value) -> Optional[int]:
    """Parse a start/end line value that may be int or '3 to 8'."""
    if value is None:
        return None
    text = re.sub(r"\s+to\s+", "–", str(value).strip())
    if "–" in text:
        text = text.split("–", 1)[0].strip()
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _source_uses_paragraph_breaks(source: str) -> bool:
    """HTML extracts join <p> tags with newlines — those are paragraphs, not editor lines."""
    source = (source or "").lower()
    return (
        source.startswith("http://")
        or source.startswith("https://")
        or source.endswith(".html")
        or source.endswith(".htm")
    )


def enrich_location_metadata_for_display(metadata: dict, page_content: str) -> dict:
    """Correct location labels using the passage text actually shown.

    Chunking is character-based, so a long HTML <p> can still be start_line==end_line
    even when the visible passage is large. Recompute the span from content newlines
    and fall back to character offsets when the passage stays on one paragraph break.
    """
    meta = dict(metadata or {})
    content = page_content or ""
    start = _parse_line_bound(meta.get("start_line"))
    end_meta = _parse_line_bound(meta.get("end_line"))
    content_breaks = content.count("\n")

    if start is not None:
        end = start + content_breaks
        if end_meta is not None:
            end = max(end, end_meta)
        meta["start_line"] = f"{start} to {end}" if end > start else start

    # Long single-break passages: keep character span so users see extent.
    if (
        content_breaks == 0
        and len(content) > 160
        and meta.get("start_index") is not None
    ):
        meta["_show_char_span"] = True
    return meta


def _format_line_range(value, *, unit: str = "paragraph") -> Optional[str]:
    """Format start_line metadata (possibly '12 to 48') for display."""
    text = re.sub(r"\s+to\s+", "–", str(value).strip())
    singular, plural = unit, unit + "s"
    if "–" in text:
        left, _, right = text.partition("–")
        left, right = left.strip(), right.strip()
        if left == right:
            return f"{singular}: {left}"
        return f"{plural}: {left}–{right}"
    return f"{singular}: {text}"


def _format_char_span(value) -> Optional[str]:
    text = re.sub(r"\s+to\s+", "–", str(value).strip())
    if _metadata_value_is_empty(text):
        return None
    if "–" in text:
        left, _, right = text.partition("–")
        left, right = left.strip(), right.strip()
        if left == right:
            return f"character offset: {left}"
        return f"character range: {left}–{right}"
    return f"character offset: {text}"


def _format_metadata_item(
    key: str, value, *, unit: str = "paragraph", show_char_span: bool = False
) -> Optional[str]:
    """Return a single 'label: value' fragment, or None to omit."""
    if key in {"page_section", "end_line", "_show_char_span"}:
        return None
    if key == "start_index" and not show_char_span:
        return None
    if _metadata_value_is_empty(value) and key != "start_index":
        return None

    text = str(value).strip() if value is not None else ""

    if key == "start_line":
        return _format_line_range(text, unit=unit)

    if key == "start_index":
        return _format_char_span(text)

    # Empty date joined as " to 12 March..." or "12 March to "
    if key == "date":
        text = re.sub(r"^\s*to\s+", "", text)
        text = re.sub(r"\s+to\s*$", "", text)
        text = re.sub(r"\s+to\s+", "–", text)
        if _metadata_value_is_empty(text):
            return None

    label = _META_DISPLAY_LABELS.get(key, key)
    return f"{label}: {text}"


def write_out_metadata_as_string(metadata_in, page_contents=None):
    """Format passage metadata for the sources panel.

    ``page_contents`` (optional, same length as metadata_in) is used to correct
    paragraph/line spans from the text actually displayed.
    """
    metadata_string = []
    for i, d in enumerate(metadata_in):
        if not isinstance(d, dict):
            continue
        content = ""
        if page_contents is not None and i < len(page_contents):
            content = page_contents[i] or ""
        d = enrich_location_metadata_for_display(d, content)

        source = str(d.get("source") or d.get("meta_url") or "")
        unit = "paragraph" if _source_uses_paragraph_breaks(source) else "line"
        show_char = bool(d.get("_show_char_span")) or (
            "start_line" not in d and "start_index" in d
        )

        parts = []
        for k, v in d.items():
            formatted = _format_metadata_item(k, v, unit=unit, show_char_span=show_char)
            if formatted:
                parts.append(formatted)
        metadata_string.append("  ".join(parts))
    return metadata_string


def generate_expanded_prompt(
    inputs: Dict[str, str],
    instruction_prompt: str,
    content_prompt: str,
    extracted_memory: list,
    vectorstore: object,
    embeddings_model: object,
    relevant_flag: bool = True,
    out_passages: int = 2,
    total_output_passage_chunks_size: int = 5,
):
    """
    Generate an expanded prompt for a language model by retrieving and formatting relevant document passages.

    Args:
        inputs (Dict[str, str]): Dictionary containing the user's question and chat history.
        instruction_prompt (str): The instruction prompt template to use for the model.
        content_prompt (str): The content prompt template for formatting passages.
        extracted_memory (list): List of previous conversation memory or context.
        vectorstore (object): The vector store object used for document retrieval.
        embeddings_model (object): The embeddings model used for vector search.
        relevant_flag (bool, optional): Whether to perform relevant document retrieval. Defaults to True.
        out_passages (int, optional): Number of passages to retrieve. Defaults to 2.
        total_output_passage_chunks_size (int, optional): Number of neighboring chunks to expand for context. Defaults to 5.

    Returns:
        tuple: (instruction_prompt_out, sources_docs_content_string, new_question_kworded, scored_passages)
            instruction_prompt_out (str): The fully formatted instruction prompt for the model.
            sources_docs_content_string (str): The formatted string of source passages and metadata for user display.
            new_question_kworded (str): The (possibly keyword-adapted) user question.
            scored_passages (list[dict]): Winning passages with text, score, and source for Phoenix.
    """

    question = inputs["question"]
    chat_history = inputs["chat_history"]

    if relevant_flag:
        new_question_kworded = adapt_q_from_chat_history(
            question, chat_history, extracted_memory
        )  # new_question_keywords,
        docs_keep_as_doc, doc_df, docs_keep_out = hybrid_retrieval(
            new_question_kworded,
            vectorstore,
            embeddings_model,
            k_val=25,
            out_passages=out_passages,
            # Cosine similarity after L2-normalisation is roughly in [-1, 1]; keep
            # positively similar hits (old cut-off of 1 filtered almost everything).
            vec_score_cut_off=0.0,
            vec_weight=1,
            bm25_weight=1,
            svm_weight=1,
        )
    else:
        new_question_kworded = question
        doc_df = pd.DataFrame()
        docs_keep_as_doc = []
        docs_keep_out = []

    scored_passages: list[dict] = []
    for item in docs_keep_out or []:
        try:
            doc, score = item[0], float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        scored_passages.append(
            {
                "text": getattr(doc, "page_content", "") or "",
                "score": score,
                "source": (getattr(doc, "metadata", None) or {}).get("source", ""),
                "metadata": getattr(doc, "metadata", None) or {},
            }
        )

    if (not docs_keep_as_doc) | (doc_df.empty):
        sorry_prompt = """Respond 'Sorry, there is no relevant information to answer this question.'"""
        return (
            sorry_prompt,
            "No relevant sources found.",
            new_question_kworded,
            scored_passages,
        )

    # Expand the found passages to the neighbouring context
    if "meta_url" in doc_df.columns:
        file_type = determine_file_type(doc_df["meta_url"][0])
    else:
        file_type = determine_file_type(doc_df["source"][0])

    # Only expand passages if not tabular data
    if (file_type != ".csv") & (file_type != ".xlsx"):
        docs_keep_as_doc, doc_df = get_expanded_passages(
            vectorstore, docs_keep_out, width=total_output_passage_chunks_size
        )

    # Build up sources content to add to user display
    # Use passage text to correct paragraph/line spans (HTML <p> breaks != editor lines).
    page_contents = doc_df["page_content"].tolist()
    doc_df["meta_clean"] = write_out_metadata_as_string(
        doc_df["metadata"], page_contents=page_contents
    )

    # Remove meta text from the page content if it already exists there
    doc_df["page_content_no_meta"] = doc_df.apply(
        lambda row: row["page_content"].replace(row["meta_clean"] + ". ", ""), axis=1
    )
    doc_df["content_meta"] = (
        doc_df["meta_clean"].astype(str)
        + ".<br><br>"
        + doc_df["page_content_no_meta"].astype(str)
    )

    # modified_page_content = [f" Document {i+1} - {word}" for i, word in enumerate(doc_df['page_content'])]
    modified_page_content = [
        f" Document {i+1} - {word}" for i, word in enumerate(doc_df["content_meta"])
    ]
    docs_content_string = "<br><br>".join(modified_page_content)

    sources_docs_content_string = "<br><br>".join(
        doc_df["content_meta"]
    )  # .replace("  "," ")#.strip()

    instruction_prompt_out = instruction_prompt.replace(
        "{question}", new_question_kworded
    ).replace("{summaries}", docs_content_string)

    return (
        instruction_prompt_out,
        sources_docs_content_string,
        new_question_kworded,
        scored_passages,
    )


def create_full_prompt(
    user_input: str,
    history: list[dict],
    extracted_memory: str,
    vectorstore: object,
    embeddings_model: object,
    model_type: str,
    out_passages: list[str],
    api_key: str = "",
    bedrock_session_id: str = "",
    session_hash: str = "",
    relevant_flag: bool = True,
):

    if "gemini" in model_type and not GEMINI_API_KEY and not api_key:
        raise Exception(
            "Gemini model selected but no API key found. Please enter an API key on the Advanced settings page."
        )

    # if chain_agent is None:
    #    history.append((user_input, "Please click the button to submit the Huggingface API key before using the chatbot (top right)"))
    #    return history, history, "", ""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")

    history = history or []
    topic_out = extracted_memory or ""
    session_out = bedrock_session_id or ""

    if not user_input.strip():
        user_input = "No user input found"
        relevant_flag = False
    else:
        relevant_flag = True

    from tools.phoenix_tracing import (
        chat_span,
        set_retrieval_documents,
        set_span_attr,
        set_span_output,
    )

    backend = "bedrock_kb" if USE_BEDROCK_KB == "1" else "local_faiss"
    traced_model = (
        f"AWS Bedrock KB ({BEDROCK_MODEL_ID})" if USE_BEDROCK_KB == "1" else model_type
    )
    with chat_span(
        "chat.retrieve",
        kind="RETRIEVER" if backend == "local_faiss" else "CHAIN",
        session_hash=session_hash,
        input_value=user_input,
        attributes={
            "qa.backend": backend,
            "qa.model_type": traced_model,
            "qa.user_query": user_input,
            "qa.bedrock_session_id": session_out or None,
        },
    ) as span:
        # Bedrock Knowledge Base: retrieve + generate in one call; skip local FAISS RAG.
        if USE_BEDROCK_KB == "1":
            from tools.bedrock_kb import get_rag_response

            if not relevant_flag:
                history.append(
                    {
                        "metadata": None,
                        "options": None,
                        "role": "user",
                        "content": user_input,
                    }
                )
                set_span_output(span, "")
                return (
                    history,
                    "No relevant source paragraphs currently loaded",
                    "",
                    False,
                    session_out,
                    topic_out,
                )

            result = get_rag_response(user_input, session_id=session_out or None)
            session_out = result.get("sessionId", "") or ""
            topic_out = result.get("topic", "General Enquiry")
            docs_content_string = result.get("sources_html", "")
            # Answer text is passed through instruction_prompt_out for the streamer step.
            instruction_prompt_out = result.get("response", "")

            set_span_output(span, instruction_prompt_out)
            set_span_attr(span, "qa.topic", topic_out)
            set_span_attr(span, "qa.citations_count", result.get("citations_count", 0))
            set_span_attr(span, "qa.passages_count", result.get("passages_count", 0))
            set_span_attr(span, "qa.top_score", result.get("top_score"))
            scores = result.get("retrieval_scores") or []
            if scores:
                set_span_attr(
                    span,
                    "qa.retrieval_scores",
                    ",".join(f"{s:.4f}" for s in scores),
                )
            set_retrieval_documents(span, result.get("passages") or [])
            set_span_attr(span, "qa.guardrail", result.get("guardrail", False))
            set_span_attr(span, "qa.bedrock_session_id", session_out or None)
            if result.get("retrieve_error"):
                set_span_attr(span, "qa.retrieve_error", result["retrieve_error"])

            history.append(
                {
                    "metadata": None,
                    "options": None,
                    "role": "user",
                    "content": user_input,
                }
            )
            return (
                history,
                docs_content_string,
                instruction_prompt_out,
                True,
                session_out,
                topic_out,
            )

        # Create instruction prompt
        instruction_prompt, content_prompt = base_prompt_templates(
            model_type=model_type
        )

        (
            instruction_prompt_out,
            docs_content_string,
            new_question_kworded,
            scored_passages,
        ) = generate_expanded_prompt(
            {"question": user_input, "chat_history": history},  # vectorstore,
            instruction_prompt,
            content_prompt,
            extracted_memory,
            vectorstore,
            embeddings_model,
            relevant_flag,
            out_passages,
        )

        retrieval_scores = [
            p["score"]
            for p in scored_passages
            if isinstance(p.get("score"), (int, float))
        ]
        set_span_output(span, docs_content_string)
        set_span_attr(span, "qa.expanded_query", new_question_kworded)
        set_span_attr(span, "qa.relevant", relevant_flag)
        set_span_attr(span, "qa.passages_count", len(scored_passages))
        if retrieval_scores:
            set_span_attr(span, "qa.top_score", retrieval_scores[0])
            set_span_attr(
                span,
                "qa.retrieval_scores",
                ",".join(f"{s:.4f}" for s in retrieval_scores),
            )
        set_retrieval_documents(span, scored_passages)

        history.append(
            {"metadata": None, "options": None, "role": "user", "content": user_input}
        )

        return (
            history,
            docs_content_string,
            instruction_prompt_out,
            relevant_flag,
            session_out,
            topic_out,
        )


def call_aws_bedrock(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    model_choice: str,
) -> ResponseObject:
    """
    Send a request to an AWS Bedrock model via the Converse API.

    Works across Anthropic, Amazon Nova, and other Bedrock providers listed in AWS_MODELS.
    Returns a ResponseObject with response text and usage metadata.
    """
    inference_config = {
        "maxTokens": max_tokens,
        "topP": 0.999,
        "temperature": temperature,
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }
    ]

    system_prompt_list = [{"text": system_prompt}]

    api_response = bedrock_runtime.converse(
        modelId=model_choice,
        messages=messages,
        system=system_prompt_list,
        inferenceConfig=inference_config,
    )

    output_message = api_response["output"]["message"]

    if "reasoningContent" in output_message["content"][0]:
        text = output_message["content"][1]["text"]
    else:
        text = output_message["content"][0]["text"]

    usage = api_response["usage"]
    print("Metadata:", usage)

    return ResponseObject(text=text, usage_metadata=usage)


def call_aws_claude(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    model_choice: str,
) -> ResponseObject:
    """Backward-compatible alias for call_aws_bedrock."""
    return call_aws_bedrock(
        prompt, system_prompt, temperature, max_tokens, model_choice
    )


def construct_gemini_generative_model(
    in_api_key: str,
    temperature: float,
    model_choice: str,
    system_prompt: str,
    max_tokens: int,
    random_seed: int = None,
) -> Tuple[object, dict]:
    """
    Constructs a Client for Gemini API calls using the new google.genai package.

    Parameters:
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model, controlling the randomness of the output.
    - model_choice (str): The choice of model to use for generation.
    - system_prompt (str): The system prompt to guide the generation.
    - max_tokens (int): The maximum number of tokens to generate.
    - random_seed (int, optional): Random seed for reproducibility.

    Returns:
    - Tuple[object, dict]: A tuple containing the constructed Client and its configuration.
    """
    # Construct a Client for the new API
    try:
        if in_api_key:
            # print("Getting API key from textbox")
            api_key = in_api_key
            client = ai.Client(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            # print("Searching for API key in environmental variables")
            api_key = os.environ["GOOGLE_API_KEY"]
            client = ai.Client(api_key=api_key)
        else:
            print("No API key found")
            raise gr.Error("No API key found.")
    except Exception as e:
        print(e)
        raise

    # Create config with optional random_seed
    config_kwargs = {"temperature": temperature, "max_output_tokens": max_tokens}
    if random_seed is not None:
        config_kwargs["seed"] = random_seed
    config = types.GenerateContentConfig(**config_kwargs)

    print("model_choice:", model_choice)

    return client, config


# Function to send a request and update history
def send_request(
    prompt: str,
    conversation_history: List[dict],
    model: object,
    config: dict,
    model_choice: str,
    system_prompt: str,
    temperature: float,
    progress=Progress(track_tqdm=True),
) -> Tuple[str, List[dict]]:
    """
    This function sends a request to a language model with the given prompt, conversation history, model configuration, model choice, system prompt, and temperature.
    It constructs the full prompt by appending the new user prompt to the conversation history, generates a response from the model, and updates the conversation history with the new prompt and response.
    If the model choice is an AWS Bedrock model, it calls call_aws_bedrock; Gemini uses the Google API; otherwise raises.
    The function returns the response text and the updated conversation history.
    """
    # Constructing the full prompt from the conversation history
    full_prompt = "Conversation history:\n"

    for entry in conversation_history:
        role = entry[
            "role"
        ].capitalize()  # Assuming the history is stored with 'role' and 'content'
        message = " ".join(entry["parts"])  # Combining all parts of the message
        full_prompt += f"{role}: {message}\n"

    # Adding the new user prompt
    full_prompt += f"\nUser: {prompt}"

    # Print the full prompt for debugging purposes
    # print("full_prompt:", full_prompt)

    # Generate the model's response — AWS before Gemini so Bedrock Gemma IDs are not misrouted
    if model_choice in AWS_MODELS:
        try:
            print("Calling AWS Bedrock model:", model_choice)
            response = call_aws_bedrock(
                prompt, system_prompt, temperature, max_tokens, model_choice
            )
        except Exception as e:
            # If fails, try again after x seconds in case there is a throttle limit
            print(e)
            try:
                out_message = "API limit hit - waiting 30 seconds to retry."
                print(out_message)
                progress(0.5, desc=out_message)
                time.sleep(30)
                response = call_aws_bedrock(
                    prompt, system_prompt, temperature, max_tokens, model_choice
                )

            except Exception as e:
                print(e)
                return "", conversation_history
    elif "gemini" in model_choice:
        try:
            # New API: client.models.generate_content instead of model.generate_content
            gemini_response = model.models.generate_content(
                model=model_choice, contents=full_prompt, config=config
            )
            # Wrap response in ResponseObject for backwards compatibility
            usage_metadata = {}
            if hasattr(gemini_response, "usage_metadata"):
                usage_metadata = gemini_response.usage_metadata
            elif hasattr(gemini_response, "usage"):
                usage_metadata = gemini_response.usage
            response = ResponseObject(
                text=gemini_response.text, usage_metadata=usage_metadata
            )
        except Exception as e:
            # If fails, try again after 10 seconds in case there is a throttle limit
            print(e)
            try:
                print("Calling Gemini model")
                out_message = "API limit hit - waiting 30 seconds to retry."
                print(out_message)
                progress(0.5, desc=out_message)
                time.sleep(30)
                gemini_response = model.models.generate_content(
                    model=model_choice, contents=full_prompt, config=config
                )
                # Wrap response in ResponseObject for backwards compatibility
                usage_metadata = {}
                if hasattr(gemini_response, "usage_metadata"):
                    usage_metadata = gemini_response.usage_metadata
                elif hasattr(gemini_response, "usage"):
                    usage_metadata = gemini_response.usage
                response = ResponseObject(
                    text=gemini_response.text, usage_metadata=usage_metadata
                )
            except Exception as e:
                print(e)
                return "", conversation_history
    else:
        raise Exception("Model not found")

    # Update the conversation history with the new prompt and response
    conversation_history.append(
        {"metadata": None, "options": None, "role": "user", "parts": [prompt]}
    )
    conversation_history.append(
        {
            "metadata": None,
            "options": None,
            "role": "assistant",
            "parts": [response.text],
        }
    )

    # Print the updated conversation history
    # print("conversation_history:", conversation_history)

    return response, conversation_history


def process_requests(
    prompts: List[str],
    system_prompt_with_table: str,
    conversation_history: List[dict],
    whole_conversation: List[str],
    whole_conversation_metadata: List[str],
    model: object,
    config: dict,
    model_choice: str,
    temperature: float,
    batch_no: int = 1,
    master: bool = False,
) -> Tuple[List[ResponseObject], List[dict], List[str], List[str]]:
    """
    Processes a list of prompts by sending them to the model, appending the responses to the conversation history, and updating the whole conversation and metadata.

    Args:
        prompts (List[str]): A list of prompts to be processed.
        system_prompt_with_table (str): The system prompt including a table.
        conversation_history (List[dict]): The history of the conversation.
        whole_conversation (List[str]): The complete conversation including prompts and responses.
        whole_conversation_metadata (List[str]): Metadata about the whole conversation.
        model (object): The model to use for processing the prompts.
        config (dict): Configuration for the model.
        model_choice (str): The choice of model to use.
        temperature (float): The temperature parameter for the model.
        batch_no (int): Batch number of the large language model request.
        master (bool): Is this request for the master table.

    Returns:
        Tuple[List[ResponseObject], List[dict], List[str], List[str]]: A tuple containing the list of responses, the updated conversation history, the updated whole conversation, and the updated whole conversation metadata.
    """
    responses = []
    # for prompt in prompts:

    response, conversation_history = send_request(
        prompts[0],
        conversation_history,
        model=model,
        config=config,
        model_choice=model_choice,
        system_prompt=system_prompt_with_table,
        temperature=temperature,
    )

    # print(response.text)
    # print(response.usage_metadata)
    responses.append(response)

    # Create conversation txt object
    whole_conversation.append(prompts[0])
    whole_conversation.append(response.text)

    # Create conversation metadata
    if not master:
        whole_conversation_metadata.append(
            f"Query batch {batch_no} prompt {len(responses)} metadata:"
        )
    else:
        whole_conversation_metadata.append("Query summary metadata:")

    whole_conversation_metadata.append(str(response.usage_metadata))

    return (
        responses,
        conversation_history,
        whole_conversation,
        whole_conversation_metadata,
    )


def produce_streaming_answer_chatbot(
    history: list,
    full_prompt: str,
    model_type: str,
    temperature: float = temperature,
    relevant_query_bool: bool = True,
    chat_history: list[dict] = [
        {"metadata": None, "options": None, "role": "user", "content": ""}
    ],
    in_api_key: str = GEMINI_API_KEY,
    session_hash: str = "",
    user_query: str = "",
    max_new_tokens: int = max_new_tokens,
    sample: bool = sample,
    repetition_penalty: float = repetition_penalty,
    top_p: float = top_p,
    top_k: float = top_k,
    max_tokens: int = max_tokens,
):
    """Stream chat answers; emit a Phoenix LLM span when tracing is enabled."""
    from tools.phoenix_tracing import (
        end_chat_span,
        set_span_output,
        start_chat_span,
        tracing_initialized,
    )

    kwargs = dict(
        history=history,
        full_prompt=full_prompt,
        model_type=model_type,
        temperature=temperature,
        relevant_query_bool=relevant_query_bool,
        chat_history=chat_history,
        in_api_key=in_api_key,
        max_new_tokens=max_new_tokens,
        sample=sample,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    if not tracing_initialized():
        yield from _produce_streaming_answer_chatbot_impl(**kwargs)
        return

    backend = "bedrock_kb" if USE_BEDROCK_KB == "1" else "local_generate"
    traced_model = (
        f"AWS Bedrock KB ({BEDROCK_MODEL_ID})" if USE_BEDROCK_KB == "1" else model_type
    )
    prompt_preview = full_prompt if isinstance(full_prompt, str) else str(full_prompt)
    # Prefer the raw Gradio message; fall back to last user turn in history.
    raw_query = (user_query or "").strip()
    if not raw_query and chat_history:
        for turn in reversed(chat_history):
            if isinstance(turn, dict) and turn.get("role") == "user":
                content = turn.get("content")
                if isinstance(content, str):
                    raw_query = content.strip()
                elif isinstance(content, list):
                    raw_query = " ".join(
                        (
                            str(part.get("text", part))
                            if isinstance(part, dict)
                            else str(part)
                        )
                        for part in content
                    ).strip()
                break

    # start_span (not as_current): Gradio yields across contexts/threads.
    span = start_chat_span(
        "chat.generate",
        kind="LLM",
        session_hash=session_hash,
        input_value=prompt_preview,
        attributes={
            "qa.backend": backend,
            "qa.model_type": traced_model,
            "qa.user_query": raw_query or None,
        },
    )
    final_text = ""
    error: BaseException | None = None
    try:
        for hist in _produce_streaming_answer_chatbot_impl(**kwargs):
            if (
                hist
                and isinstance(hist[-1], dict)
                and hist[-1].get("role") == "assistant"
            ):
                final_text = hist[-1].get("content") or final_text
            yield hist
        set_span_output(span, final_text)
    except BaseException as exc:
        error = exc
        raise
    finally:
        end_chat_span(span, error=error)


def _produce_streaming_answer_chatbot_impl(
    history: list,
    full_prompt: str,
    model_type: str,
    temperature: float = temperature,
    relevant_query_bool: bool = True,
    chat_history: list[dict] = [
        {"metadata": None, "options": None, "role": "user", "content": ""}
    ],
    in_api_key: str = GEMINI_API_KEY,
    max_new_tokens: int = max_new_tokens,
    sample: bool = sample,
    repetition_penalty: float = repetition_penalty,
    top_p: float = top_p,
    top_k: float = top_k,
    max_tokens: int = max_tokens,
):
    # print("Model type is: ", model_type)

    # if not full_prompt.strip():
    #    if history is None:
    #        history = []

    #    return history

    history = chat_history

    if not relevant_query_bool:
        history.append(
            {
                "metadata": None,
                "options": None,
                "role": "assistant",
                "content": "No relevant query found. Please retry your question",
            }
        )

        yield history
        return

    # Bedrock KB already produced the final answer in create_full_prompt.
    if USE_BEDROCK_KB == "1":
        answer_text = full_prompt if isinstance(full_prompt, str) else str(full_prompt)
        answer_text = strip_model_artifacts(
            (answer_text or "").strip() or "No response from knowledge base.",
            finalize=True,
        )

        history.append(
            {
                "metadata": None,
                "options": None,
                "role": "assistant",
                "content": "",
            }
        )
        for char in answer_text:
            time.sleep(0.001)
            history[-1]["content"] += char
            yield history
        return

    if model_type == SMALL_MODEL_NAME:

        begin_generation_run()

        # Get the model and tokenizer, and tokenize the user text.
        model_inputs = tokenizer(
            text=full_prompt, return_tensors="pt", return_attention_mask=False
        ).to(torch_device)

        # timeout=None: we apply first-token / between-token waits ourselves below.
        streamer = TextIteratorStreamer(
            tokenizer, timeout=None, skip_prompt=True, skip_special_tokens=True
        )
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k,
            stopping_criteria=StoppingCriteriaList([_CancelOnEvent()]),
        )

        t = Thread(target=model_object.generate, kwargs=generate_kwargs, daemon=True)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS = 0
        print("-" * 4 + "Start Generation" + "-" * 4)
        print(
            f"First-token timeout: {GENERATION_FIRST_TOKEN_TIMEOUT:.0f}s; "
            f"between-token timeout: {GENERATION_TOKEN_TIMEOUT:.0f}s"
        )

        history.append(
            {"metadata": None, "options": None, "role": "assistant", "content": ""}
        )
        raw_output = ""
        # Yield immediately so the UI is not blank while waiting on first token,
        # and so Gradio can process Stop / cancel between yields.
        yield history

        try:
            for new_text in _iter_text_streamer(
                streamer,
                first_token_timeout=GENERATION_FIRST_TOKEN_TIMEOUT,
                token_timeout=GENERATION_TOKEN_TIMEOUT,
            ):
                if _generation_cancel.is_set():
                    history[-1]["content"] = strip_model_artifacts(
                        raw_output, finalize=True
                    ) or "Generation stopped."
                    yield history
                    return
                if new_text is None:
                    new_text = ""
                raw_output += new_text
                history[-1]["content"] = strip_model_artifacts(raw_output)
                NUM_TOKENS += 1
                yield history
        except TimeoutError as e:
            _generation_cancel.set()
            print(f"Generation timeout: {e}")
            history[-1]["content"] = str(e)
            yield history
            return
        except Exception as e:
            _generation_cancel.set()
            print(f"Error during text generation: {e}")
            history[-1]["content"] = (
                strip_model_artifacts(raw_output, finalize=True)
                or f"Generation failed: {e}"
            )
            yield history
            return

        if _generation_cancel.is_set():
            history[-1]["content"] = (
                strip_model_artifacts(raw_output, finalize=True)
                or "Generation stopped."
            )
            yield history
            return

        history[-1]["content"] = strip_model_artifacts(raw_output, finalize=True)
        yield history

        time_generate = time.time() - start
        print("\n")
        print("-" * 4 + "End Generation" + "-" * 4)
        print(f"Num of generated tokens: {NUM_TOKENS}")
        if NUM_TOKENS > 0 and time_generate > 0:
            print(f"Time for complete generation: {time_generate:.2f}s")
            print(f"Tokens per secound: {NUM_TOKENS/time_generate:.2f}")
            print(f"Time per token: {(time_generate/NUM_TOKENS)*1000:.2f}ms")

    elif model_type == LARGE_MODEL_NAME:
        # tokens = model.tokenize(full_prompt)

        gen_config = CtransGenGenerationConfig()
        gen_config.update_temp(temperature)

        print(vars(gen_config))

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS = 0
        print("-" * 4 + "Start Generation" + "-" * 4)

        output = model_object(full_prompt, **vars(gen_config))

        history.append(
            {"metadata": None, "options": None, "role": "assistant", "content": ""}
        )
        raw_output = ""

        for out in output:

            if (
                "choices" in out
                and len(out["choices"]) > 0
                and "text" in out["choices"][0]
            ):
                raw_output += out["choices"][0]["text"]
                history[-1]["content"] = strip_model_artifacts(raw_output)
                NUM_TOKENS += 1
                yield history
            else:
                print(f"Unexpected output structure: {out}")

        history[-1]["content"] = strip_model_artifacts(raw_output, finalize=True)
        yield history

        time_generate = time.time() - start
        print("\n")
        print("-" * 4 + "End Generation" + "-" * 4)
        print(f"Num of generated tokens: {NUM_TOKENS}")
        print(f"Time for complete generation: {time_generate:.2f}s")
        print(f"Tokens per second: {NUM_TOKENS/time_generate:.2f}")
        print(f"Time per token: {(time_generate/NUM_TOKENS)*1000:.2f}ms")

    elif model_type in AWS_MODELS:
        system_prompt = "You are answering questions from the user based on source material. Make sure to fully answer the questions with all required detail."

        if isinstance(full_prompt, str):
            full_prompt = [full_prompt]

        model = model_type
        config = {}

        (
            responses,
            summary_conversation_history,
            whole_summary_conversation,
            whole_conversation_metadata,
        ) = process_requests(
            full_prompt,
            system_prompt,
            conversation_history=[],
            whole_conversation=[],
            whole_conversation_metadata=[],
            model=model,
            config=config,
            model_choice=model_type,
            temperature=temperature,
        )

        if isinstance(responses[-1], ResponseObject):
            response_texts = [resp.text for resp in responses]
        elif "choices" in responses[-1]:
            response_texts = [resp["choices"][0]["text"] for resp in responses]
        else:
            response_texts = [resp.text for resp in responses]

        latest_response_text = response_texts[-1]
        latest_response_text = strip_model_artifacts(
            latest_response_text, finalize=True
        )

        # Update the conversation history with the new prompt and response
        clean_text = re.sub(
            r"[\n\t\r]", " ", latest_response_text
        )  # Replace newlines, tabs, and carriage returns with a space
        clean_response_text = re.sub(
            r"[^\x20-\x7E]", "", clean_text
        ).strip()  # Remove all non-ASCII printable characters

        history.append(
            {"metadata": None, "options": None, "role": "assistant", "content": ""}
        )

        for char in clean_response_text:
            time.sleep(0.001)
            history[-1]["content"] += char
            yield history

    elif "gemini" in model_type:

        if in_api_key:
            gemini_api_key = in_api_key
        elif GEMINI_API_KEY:
            gemini_api_key = GEMINI_API_KEY
        else:
            raise Exception(
                "Gemini API key not found. Please enter a key on the Advanced settings page or select another model type"
            )

        if isinstance(full_prompt, str):
            full_prompt = [full_prompt]

        system_prompt = "You are answering questions from the user based on source material. Make sure to fully answer the questions with all required detail."

        model, config = construct_gemini_generative_model(
            gemini_api_key, temperature, model_type, system_prompt, max_tokens
        )

        (
            responses,
            summary_conversation_history,
            whole_summary_conversation,
            whole_conversation_metadata,
        ) = process_requests(
            full_prompt,
            system_prompt,
            conversation_history=[],
            whole_conversation=[],
            whole_conversation_metadata=[],
            model=model,
            config=config,
            model_choice=model_type,
            temperature=temperature,
        )

        if isinstance(responses[-1], ResponseObject):
            response_texts = [resp.text for resp in responses]
        elif "choices" in responses[-1]:
            response_texts = [resp["choices"][0]["text"] for resp in responses]
        else:
            response_texts = [resp.text for resp in responses]

        latest_response_text = response_texts[-1]
        latest_response_text = strip_model_artifacts(
            latest_response_text, finalize=True
        )

        clean_text = re.sub(
            r"[\n\t\r]", " ", latest_response_text
        )  # Replace newlines, tabs, and carriage returns with a space
        clean_response_text = re.sub(
            r"[^\x20-\x7E]", "", clean_text
        ).strip()  # Remove all non-ASCII printable characters

        history.append(
            {"metadata": None, "options": None, "role": "assistant", "content": ""}
        )

        for char in clean_response_text:
            time.sleep(0.001)
            history[-1]["content"] += char
            yield history

        # print("history at end of function:", history)


# Chat helper functions


def adapt_q_from_chat_history(
    question, chat_history, extracted_memory, keyword_model=""
):  # keyword_model): # new_question_keywords,

    (
        chat_history_str,
        chat_history_first_q,
        chat_history_first_ans,
        max_memory_length,
    ) = _get_chat_history(chat_history)

    if chat_history_str:
        # Keyword extraction is now done in the add_inputs_to_history function
        # remove_q_stopwords(str(chat_history_first_q) + " " + str(chat_history_first_ans))

        new_question_kworded = (
            str(extracted_memory) + ". " + question
        )  # + " " + new_question_keywords
        # extracted_memory + " " + question

    else:
        new_question_kworded = question  # new_question_keywords

    # print("Question output is: " + new_question_kworded)

    return new_question_kworded


def determine_file_type(file_path):
    """
    Determine the file type based on its extension.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        str: File extension (e.g., '.pdf', '.docx', '.txt', '.html').
    """
    return os.path.splitext(file_path)[1].lower()


def create_doc_df(docs_keep_out):
    # Extract content and metadata from 'winning' passages.
    content = []
    meta = []
    meta_url = []
    page_section = []
    score = []

    doc_df = pd.DataFrame()

    for item in docs_keep_out:
        content.append(item[0].page_content)
        meta.append(item[0].metadata)
        meta_url.append(item[0].metadata["source"])

        file_extension = determine_file_type(item[0].metadata["source"])
        if (file_extension != ".csv") & (file_extension != ".xlsx"):
            page_section.append(item[0].metadata["page_section"])
        else:
            page_section.append("")
        score.append(item[1])

    # Create df from 'winning' passages

    doc_df = pd.DataFrame(
        list(zip(content, meta, page_section, meta_url, score)),
        columns=["page_content", "metadata", "page_section", "meta_url", "score"],
    )

    doc_df["page_content"].astype(str)
    doc_df["full_url"] = "https://" + doc_df["meta_url"]

    return doc_df


def hybrid_retrieval(
    new_question_kworded: str,
    vectorstore: FAISS,
    embeddings_model: HuggingFaceEmbeddings,
    k_val: int,
    out_passages: int,
    vec_score_cut_off: float,
    vec_weight: float,
    bm25_weight: float,
    svm_weight: float,
) -> tuple:
    """
    Perform hybrid retrieval of relevant documents based on a query using vector similarity, BM25, and SVM weights.

    Args:
        new_question_kworded (str): The keyword-adapted user query.
        vectorstore: The vectorstore object for similarity search.
        embeddings_model: The embeddings model used for vector search.
        k_val (int): Number of top documents to retrieve.
        out_passages (int): Number of passages to output.
        vec_score_cut_off (float): Similarity score threshold for filtering.
        vec_weight (float): Weight for vector similarity.
        bm25_weight (float): Weight for BM25 retrieval.
        svm_weight (float): Weight for SVM retrieval.

    Returns:
        tuple: (docs_keep_as_doc, doc_df, docs_keep_out)
            docs_keep_as_doc: List of kept document objects.
            doc_df: DataFrame of kept documents and metadata.
            docs_keep_out: List of kept (document, score) tuples.
    """

    doc_df = pd.DataFrame()

    docs = vectorstore.similarity_search_with_score(new_question_kworded, k=k_val)

    # Keep only documents with a certain score
    docs_len = [len(x[0].page_content) for x in docs]
    docs_scores = [x[1] for x in docs]

    # Only keep sources that are sufficiently relevant (i.e. similarity search score above threshold below)
    score_more_limit = pd.Series(docs_scores) > vec_score_cut_off
    docs_keep = list(compress(docs, score_more_limit))

    if not docs_keep:
        return [], pd.DataFrame(), []

    # Only keep sources that are at least 100 characters long
    length_more_limit = pd.Series(docs_len) >= 100
    docs_keep = list(compress(docs_keep, length_more_limit))

    if not docs_keep:
        return [], pd.DataFrame(), []

    docs_keep_as_doc = [x[0] for x in docs_keep]
    docs_keep_length = len(docs_keep_as_doc)

    if docs_keep_length == 1:

        content = []
        meta_url = []
        score = []

        for item in docs_keep:
            content.append(item[0].page_content)
            meta_url.append(item[0].metadata["source"])
            score.append(item[1])

        # Create df from 'winning' passages

        doc_df = pd.DataFrame(
            list(zip(content, meta_url, score)),
            columns=["page_content", "meta_url", "score"],
        )

        docs_content = doc_df["page_content"].astype(str)
        docs_url = doc_df["meta_url"]

        return docs_keep_as_doc, doc_df, docs_content, docs_url

    # Check for if more docs are removed than the desired output
    if out_passages > docs_keep_length:
        out_passages = docs_keep_length
        k_val = docs_keep_length

    vec_rank = [*range(1, docs_keep_length + 1)]
    vec_score = [(docs_keep_length / x) * vec_weight for x in vec_rank]

    print("Number of documents remaining: ", docs_keep_length)

    # 2nd level check using BM25s package to do keyword search on retrieved passages.

    content_keep = []
    for item in docs_keep:
        content_keep.append(item[0].page_content)

    # Prepare Corpus (Tokenized & Optional Stemming)
    corpus = [doc.lower() for doc in content_keep]
    # stemmer = SnowballStemmer("english", ignore_stopwords=True)  # NLTK stemming not compatible
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    # Create and Index with BM25s
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # Query Processing (Stemming applied consistently if used above)
    query_tokens = bm25s.tokenize(new_question_kworded.lower(), stemmer=stemmer)
    results, scores = retriever.retrieve(
        query_tokens, corpus=corpus, k=len(corpus)
    )  # Retrieve all docs

    for i in range(results.shape[1]):
        doc, score = results[0, i], scores[0, i]
        print(f"Rank {i+1} (score: {score:.2f}): {doc}")

    # print("BM25 results:", results)
    # print("BM25 scores:", scores)

    # Rank Calculation (Custom Logic for Your BM25 Score)
    bm25_rank = list(range(1, len(results[0]) + 1))
    # bm25_rank = results[0]#.tolist()[0]  # Since you have a single query
    bm25_score = [(docs_keep_length / (rank + 1)) * bm25_weight for rank in bm25_rank]
    # +1 to avoid division by 0 for rank 0

    # Result Ordering (Using the calculated ranks)
    pairs = list(zip(bm25_rank, docs_keep_as_doc))
    pairs.sort()
    [value for rank, value in pairs]

    # 3rd level check on retrieved docs with SVM retriever
    # Note: SVM retriever removed - using vector similarity only
    # If svm_weight > 0, we'll use a simple ranking based on vector similarity
    svm_rank = []
    svm_score = []

    if svm_weight > 0:
        # Use vector similarity ranking as a proxy for SVM ranking
        # This maintains the same interface but uses vector scores
        for i, vec_item in enumerate(docs_keep):
            # Use inverse rank (lower rank = higher score)
            rank = i + 1
            svm_rank.append(rank)
            svm_score.append((docs_keep_length / rank) * svm_weight)
    else:
        # If svm_weight is 0, set all scores to 0
        svm_rank = [0] * docs_keep_length
        svm_score = [0.0] * docs_keep_length

    ## Calculate final score based on ranking methods (vector, BM25, and optionally SVM)
    # Ensure all lists have the same length
    min_len = min(len(vec_score), len(bm25_score), len(svm_score))
    final_score = [
        a + b + c
        for a, b, c in zip(
            vec_score[:min_len], bm25_score[:min_len], svm_score[:min_len]
        )
    ]
    final_rank = [sorted(final_score, reverse=True).index(x) + 1 for x in final_score]
    # Force final_rank to increment by 1 each time
    final_rank = list(pd.Series(final_rank).rank(method="first"))

    # print("final rank: " + str(final_rank))
    # print("out_passages: " + str(out_passages))

    best_rank_index_pos = []

    for x in range(1, out_passages + 1):
        try:
            best_rank_index_pos.append(final_rank.index(x))
        except IndexError:  # catch the error
            pass

    # Adjust best_rank_index_pos to

    pd.Series(best_rank_index_pos)

    docs_keep_out = [docs_keep[i] for i in best_rank_index_pos]

    # Keep only 'best' options
    docs_keep_as_doc = [x[0] for x in docs_keep_out]

    # Make df of best options
    doc_df = create_doc_df(docs_keep_out)

    return docs_keep_as_doc, doc_df, docs_keep_out


def get_expanded_passages(vectorstore, docs, width):
    """
    Extracts expanded passages based on given documents and a width for context.

    Parameters:
    - vectorstore: The primary data source.
    - docs: List of documents to be expanded.
    - width: Number of documents to expand around a given document for context.

    Returns:
    - expanded_docs: List of expanded Document objects.
    - doc_df: DataFrame representation of expanded_docs.
    """

    from collections import defaultdict

    def get_docs_from_vstore(vectorstore):
        vector = vectorstore.docstore._dict
        return list(vector.items())

    def extract_details(docs_list):
        docs_list_out = [tup[1] for tup in docs_list]
        content = [doc.page_content for doc in docs_list_out]
        meta = [doc.metadata for doc in docs_list_out]
        return "".join(content), meta[0], meta[-1]

    def get_parent_content_and_meta(vstore_docs, width, target):
        # target_range = range(max(0, target - width), min(len(vstore_docs), target + width + 1))
        target_range = range(
            max(0, target), min(len(vstore_docs), target + width + 1)
        )  # Now only selects extra passages AFTER the found passage
        parent_vstore_out = [vstore_docs[i] for i in target_range]

        content_str_out, meta_first_out, meta_last_out = [], [], []
        for _ in parent_vstore_out:
            content_str, meta_first, meta_last = extract_details(parent_vstore_out)
            content_str_out.append(content_str)
            meta_first_out.append(meta_first)
            meta_last_out.append(meta_last)
        return content_str_out, meta_first_out, meta_last_out

    def merge_dicts_except_source(d1, d2):
        """Merge first/last chunk metadata for an expanded passage.

        Empty values are skipped. Line numbers become a start–end range using
        the first chunk's start_line and the last chunk's end_line when present.
        """
        merged = {}
        keys = set(d1) | set(d2)

        # Prefer an explicit line span for expanded passages.
        start_line = d1.get("start_line")
        end_line = d2.get("end_line")
        if end_line is None:
            end_line = d2.get("start_line")
        if start_line is None:
            start_line = d2.get("start_line")
        if start_line is not None and str(start_line).strip() != "":
            if end_line is not None and str(end_line).strip() != "":
                if str(start_line) == str(end_line):
                    merged["start_line"] = start_line
                else:
                    merged["start_line"] = f"{start_line} to {end_line}"
            else:
                merged["start_line"] = start_line

        for key in keys:
            if key in {"start_line", "end_line"}:
                continue
            if key == "source":
                merged[key] = (
                    d1.get(key) if d1.get(key) not in (None, "") else d2.get(key)
                )
                continue

            v1, v2 = d1.get(key), d2.get(key)
            v1_empty = v1 is None or str(v1).strip() == ""
            v2_empty = v2 is None or str(v2).strip() == ""

            if v1_empty and v2_empty:
                continue
            if v1_empty:
                merged[key] = v2
                continue
            if v2_empty or str(v1) == str(v2):
                merged[key] = v1
                continue
            merged[key] = f"{v1} to {v2}"
        return merged

    def merge_two_lists_of_dicts(list1, list2):
        return [merge_dicts_except_source(d1, d2) for d1, d2 in zip(list1, list2)]

    # Step 1: Filter vstore_docs
    vstore_docs = get_docs_from_vstore(vectorstore)
    doc_sources = {doc.metadata["source"] for doc, _ in docs}
    vstore_docs = [
        (k, v) for k, v in vstore_docs if v.metadata.get("source") in doc_sources
    ]

    # Step 2: Group by source and proceed
    vstore_by_source = defaultdict(list)
    for k, v in vstore_docs:
        vstore_by_source[v.metadata["source"]].append((k, v))

    expanded_docs = []
    for doc, score in docs:
        search_source = doc.metadata["source"]

        # if file_type == ".csv" | file_type == ".xlsx":
        #     content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], 0, search_index)

        # else:
        search_section = doc.metadata["page_section"]
        parent_vstore_meta_section = [
            doc.metadata["page_section"] for _, doc in vstore_by_source[search_source]
        ]
        search_index = (
            parent_vstore_meta_section.index(search_section)
            if search_section in parent_vstore_meta_section
            else -1
        )

        content_str, meta_first, meta_last = get_parent_content_and_meta(
            vstore_by_source[search_source], width, search_index
        )
        meta_full = merge_two_lists_of_dicts(meta_first, meta_last)

        expanded_doc = (
            Document(page_content=content_str[0], metadata=meta_full[0]),
            score,
        )
        expanded_docs.append(expanded_doc)

    doc_df = pd.DataFrame()

    doc_df = create_doc_df(
        expanded_docs
    )  # Assuming you've defined the 'create_doc_df' function elsewhere

    return expanded_docs, doc_df


def _message_content_to_str(content) -> str:
    """Normalize Gradio chatbot message content (str or list of blocks) to a string."""
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if isinstance(content, str):
        return content
    return str(content) if content else ""


def highlight_found_text(
    chat_history: list[dict],
    source_texts: list[dict],
    min_match_words: int = hlt_min_match_words,
) -> str:
    """
    Highlight source passages that also appear in the latest assistant reply.

    Only contiguous sequences of at least ``min_match_words`` complete words
    are highlighted. Partial-word / short fragment matches are ignored.
    """

    def extract_text_from_input(text, i=0):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[i][0].replace("  ", " ").strip()
        else:
            return ""

    response_content = next(
        (
            entry["content"]
            for entry in reversed(chat_history)
            if entry.get("role") == "assistant"
        ),
        "",
    )
    response_text = _message_content_to_str(response_content)
    source_texts = extract_text_from_input(source_texts)

    if not response_text or not source_texts:
        return source_texts

    # Complete words only (no mid-token fragments)
    word_re = re.compile(r"\b[\w']+\b", re.UNICODE)
    response_words = [m.group().lower() for m in word_re.finditer(response_text)]
    if len(response_words) < min_match_words:
        return source_texts

    # All consecutive word n-grams from the reply that meet the minimum length
    response_ngrams: set[tuple[str, ...]] = set()
    for n in range(min_match_words, len(response_words) + 1):
        for i in range(len(response_words) - n + 1):
            response_ngrams.add(tuple(response_words[i : i + n]))

    source_matches = list(word_re.finditer(source_texts))
    source_words_lower = [m.group().lower() for m in source_matches]
    n_source = len(source_words_lower)

    # Greedily mark longest qualifying word sequences in the sources
    matched = [False] * n_source
    i = 0
    while i < n_source:
        best_len = 0
        max_len = min(n_source - i, len(response_words))
        for length in range(max_len, min_match_words - 1, -1):
            if tuple(source_words_lower[i : i + length]) in response_ngrams:
                best_len = length
                break
        if best_len:
            for j in range(i, i + best_len):
                matched[j] = True
            i += best_len
        else:
            i += 1

    if not any(matched):
        return source_texts

    # Collapse matched word runs into character spans (keep intervening punctuation)
    combined_positions = []
    run_start = None
    for idx, is_match in enumerate(matched):
        if is_match and run_start is None:
            run_start = idx
        elif not is_match and run_start is not None:
            if idx - run_start >= min_match_words:
                combined_positions.append(
                    (source_matches[run_start].start(), source_matches[idx - 1].end())
                )
            run_start = None
    if run_start is not None and n_source - run_start >= min_match_words:
        combined_positions.append(
            (source_matches[run_start].start(), source_matches[n_source - 1].end())
        )

    pos_tokens = []
    prev_end = 0
    for start, end in combined_positions:
        pos_tokens.append(source_texts[prev_end:start])
        pos_tokens.append(
            '<mark style="color:black;">' + source_texts[start:end] + "</mark>"
        )
        prev_end = end
    pos_tokens.append(source_texts[prev_end:])

    return "".join(pos_tokens)


# # Chat history functions


def clear_chat(
    chat_history_state, sources, chat_message, current_topic, bedrock_session_id=""
):
    chat_history_state = None
    sources = ""
    chat_message = None
    current_topic = ""
    bedrock_session_id = ""

    return chat_history_state, sources, chat_message, current_topic, bedrock_session_id


def _get_chat_history(
    chat_history: List[Tuple[str, str]], max_memory_length: int = max_memory_length
):  # Limit to last x interactions only

    if (not chat_history) | (max_memory_length == 0):
        chat_history = []

    if len(chat_history) > max_memory_length:
        chat_history = chat_history[-max_memory_length:]

    # print(chat_history)

    first_q = ""
    first_ans = ""
    for human_s, ai_s in chat_history:
        first_q = human_s
        first_ans = ai_s

        # print("Text to keyword extract: " + first_q + " " + first_ans)
        break

    conversation = ""
    for human_s, ai_s in chat_history:
        human = "Human: " + human_s
        ai = "Assistant: " + ai_s
        conversation += "\n" + "\n".join([human, ai])

    return conversation, first_q, first_ans, max_memory_length


def add_inputs_answer_to_history(user_message, history, current_topic):

    if history is None:
        history = [("", "")]

    # Bedrock KB topic comes from citation filenames; keep it instead of KeyBERT.
    if USE_BEDROCK_KB == "1":
        return history, current_topic

    # history.append((user_message, [-1]))

    (
        chat_history_str,
        chat_history_first_q,
        chat_history_first_ans,
        max_memory_length,
    ) = _get_chat_history(history)

    # Only get the keywords for the first question and response, or do it every time if over 'max_memory_length' responses in the conversation
    if (len(history) == 1) | (len(history) > max_memory_length):

        # print("History after appending is:")
        # print(history)

        first_q_and_first_ans = (
            str(chat_history_first_q) + " " + str(chat_history_first_ans)
        )
        # ner_memory = remove_q_ner_extractor(first_q_and_first_ans)
        keywords = keybert_keywords(first_q_and_first_ans, n=8, kw_model=kw_model)
        # keywords.append(ner_memory)

        # Remove duplicate words while preserving order
        ordered_tokens = set()
        result = []
        for word in keywords:
            if word not in ordered_tokens:
                ordered_tokens.add(word)
                result.append(word)

        extracted_memory = " ".join(result)

    else:
        extracted_memory = current_topic

    print("Extracted memory is:")
    print(extracted_memory)

    return history, extracted_memory


# Keyword functions


def remove_q_stopwords(
    question,
):  # Remove stopwords from question. Not used at the moment
    # Prepare keywords from question by removing stopwords
    text = question.lower()

    # Remove numbers
    text = re.sub("[0-9]", "", text)

    tokenizer = RegexpTokenizer(r"\w+")
    text_tokens = tokenizer.tokenize(text)
    # text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if word not in stopwords]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in tokens_without_sw:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)

    new_question_keywords = " ".join(result)
    return new_question_keywords


def remove_q_ner_extractor(question):

    predict_out = ner_model.predict(question)
    predict_tokens = [
        " ".join(v for k, v in d.items() if k == "span") for d in predict_out
    ]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in predict_tokens:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)

    new_question_keywords = " ".join(result).lower()
    return new_question_keywords


def apply_lemmatize(text, wnl=WordNetLemmatizer()):

    def prep_for_lemma(text):

        # Remove numbers
        text = re.sub("[0-9]", "", text)
        print(text)

        tokenizer = RegexpTokenizer(r"\w+")
        text_tokens = tokenizer.tokenize(text)
        # text_tokens = word_tokenize(text)

        return text_tokens

    tokens = prep_for_lemma(text)

    def lem_word(word):

        if len(word) > 3:
            out_word = wnl.lemmatize(word)
        else:
            out_word = word

        return out_word

    return [lem_word(token) for token in tokens]


def keybert_keywords(text, n, kw_model):
    tokens_lemma = apply_lemmatize(text)
    lemmatised_text = " ".join(tokens_lemma)

    keywords_text = KeyBERT(model=kw_model).extract_keywords(
        lemmatised_text, stop_words="english", top_n=n, keyphrase_ngram_range=(1, 1)
    )
    keywords_list = [item[0] for item in keywords_text]

    return keywords_list


# Gradio functions
def turn_off_interactivity():
    return gr.Textbox(interactive=False), gr.Button(interactive=False)


def restore_interactivity():
    return gr.Textbox(interactive=True), gr.Button(interactive=True)


def update_message(dropdown_value):
    return gr.Textbox(value=dropdown_value)


def hide_block():
    return gr.Radio(visible=False)


# Vote function


def vote(
    data: gr.LikeData,
    chat_history: list[dict],
    instruction_prompt_out: str,
    model_type: str,
    feedback_folder: str = FEEDBACK_LOGS_FOLDER,
):

    query_text = _message_content_to_str(
        next(
            (
                entry["content"]
                for entry in reversed(chat_history)
                if entry.get("role") == "user"
            ),
            "",
        )
    )

    response_text = _message_content_to_str(
        next(
            (
                entry["content"]
                for entry in reversed(chat_history)
                if entry.get("role") == "assistant"
            ),
            "",
        )
    )

    chat_history_latest = query_text + " - " + response_text

    if isinstance(data.value, list):
        chosen_response = data.value[-1]
    else:
        chosen_response = data.value

    response_df = pd.DataFrame(
        data={
            "thumbs_up": data.liked,
            "chosen_response": chosen_response,
            "input_prompt": instruction_prompt_out,
            "chat_history": chat_history_latest,
            "model_type": model_type,
            "date_time": pd.Timestamp.now(),
        },
        index=[0],
    )

    if data.liked:
        print("You upvoted this response:", chosen_response)

    else:
        print("You downvoted this response:", chosen_response)

    output_data_path = feedback_folder + "thumbs_up_down_data.csv"

    if os.path.isfile(output_data_path):
        existing_thumbs_down_df = pd.read_csv(output_data_path)
        thumbs_down_df_concat = pd.concat(
            [existing_thumbs_down_df, response_df], ignore_index=True
        ).drop("Unnamed: 0", axis=1, errors="ignore")
        thumbs_down_df_concat.to_csv(output_data_path)
    else:
        response_df.to_csv(output_data_path)

    return output_data_path
