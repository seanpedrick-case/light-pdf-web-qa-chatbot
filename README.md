---
title: Light PDF web QA chatbot
emoji: 🌍
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: true
license: mit
short_description: Chat with websites and documents using small language models
---

Chat with a pdf file or web page using a small language model (Gemma 3 1B) through a Gradio interface. Quick responses even just using CPU. Default dataset is the [user guide for my Redaction app](https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html), which you can try out [here](https://huggingface.co/spaces/seanpedrickcase/document_redaction) for basic usage, or [here](https://huggingface.co/spaces/seanpedrickcase/document_redaction_vlm) for GPU-enabled features

## AWS Bedrock Knowledge Base mode

Set `USE_BEDROCK_KB=1` to bypass local FAISS embedding/retrieval and use Bedrock Agent Runtime against an existing Knowledge Base. **KB source files and ingest prep are maintained outside this repo**; configure env vars here to match your metadata schema.

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_BEDROCK_KB` | `0` | Enable Bedrock KB backend |
| `KNOWLEDGE_BASE_ID` | _(empty)_ | Bedrock Knowledge Base ID (required when enabled) |
| `BEDROCK_MODEL_ID` | `amazon.nova-pro-v1:0` | Foundation model for retrieve_and_generate |
| `GUARDRAIL_ID` | _(empty)_ | Optional Bedrock Guardrail ID |
| `GUARDRAIL_VERSION` | _(empty)_ | Optional Guardrail version |
| `USE_KB_METADATA_FILTER` | `0` | LLM-derived metadata filter before each KB query |
| `KB_FILTER_MODEL_ID` | `amazon.nova-lite-v1:0` | Converse model for filter JSON (falls back to `BEDROCK_MODEL_ID` if empty) |
| `KB_METADATA_FILTER_KEYS` | _(empty)_ | Comma-separated metadata keys from your KB schema |
| `KB_DATE_METADATA_KEYS` | _(empty)_ | Raw date keys (rewritten/dropped per date strategy) |
| `KB_YEAR_METADATA_KEYS` | _(empty)_ | Calendar-year string columns for `year_in` |
| `KB_YEAR_COLUMNS_INDEXED` | `1` | Set `0` until `Year*` metadata exists on synced chunks (else `in` on year returns zero hits) |
| `KB_DATE_TO_YEAR_MAP` | _(empty)_ | `SourceDate:YearKey` pairs for `year_in` |
| `KB_METADATA_EXACT_EQUAL_KEYS` | `DocumentId` | Keys that keep `equals` |
| `KB_METADATA_IN_KEYS` | _(empty)_ | Optional restrict `in` to these keys |
| `KB_FILTER_PROMPT_PATH` | _(empty)_ | Custom prompt (`config/kb_filter_prompt.example.txt`) |
| `KB_GROUP_FILTER_KEY` | _(empty)_ | Allow-listed group metadata key |
| `KB_GROUP_OPTIONS_PATH` | _(empty)_ | Group labels file (e.g. local `config/kb_filter_groups.txt`, gitignored) |
| `KB_SOURCE_LABEL_METADATA_KEYS` | `DocumentId,title,description,name` | Sources panel chunk headers |
| `KB_DATE_FILTER_STRATEGY` | `semantic_only` | `semantic_only`, `year_in`, `iso_range`, `numeric_yyyymmdd` |
| `AWS_REGION` / `AWS_DEFAULT_REGION` | _(empty; falls back to `eu-west-2`)_ | Region for the agent-runtime client and model ARN |

With `USE_KB_METADATA_FILTER=1`, allow **Converse** on `KB_FILTER_MODEL_ID`. See [config/bedrock_metadata_filtering.md](config/bedrock_metadata_filtering.md).

The sources accordion uses Bedrock **Retrieve** results (ranked by `score`). Each chunk is collapsible (passage text only; metadata not inlined). Expand **Bedrock Knowledge Base retrieval** for filters, chunk counts, and outcomes.

## Optional Jina reranker (local FAISS retrieval)

After vector + BM25 hybrid fusion, you can rerank the candidate passages with [jinaai/jina-reranker-v3.5](https://huggingface.co/jinaai/jina-reranker-v3.5) via Transformers (`trust_remote_code=True`). This does not apply when `USE_BEDROCK_KB=1`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_RERANKER` | `False` | Turn on listwise reranking after hybrid retrieval |
| `RERANKER_MODEL_NAME` | `jinaai/jina-reranker-v3.5` | Hugging Face model id |
| `PRELOAD_RERANKER` | `False` | Load the reranker at app startup (else first chat query) |
| `SHOW_RETRIEVAL_DETAILS` | `True` | Prepend vector / hybrid / rerank score breakdown to the sources panel |

Example in `config/app_config.env`:

```
ENABLE_RERANKER=True
PRELOAD_RERANKER=True
```

The reranker is ~0.6B parameters; expect extra RAM (~1–2GB) and slower retrieval on CPU. Set `HF_TOKEN` if the Hub requires authentication. Source passage scores shown in the UI / Phoenix use the reranker `relevance_score` when reranking is enabled.

With `SHOW_RETRIEVAL_DETAILS=True`, each answer updates the closed **Retrieval diagnostics** accordion on the Chatbot tab (Markdown): FAISS cosine shortlist, hybrid fusion ranks/scores, and reranker scores when enabled. The **Source paragraphs** panel shows passage text only. Set `LOGGING=True` for JSON in the console and BM25 rank prints during retrieval.

**License:** `jina-reranker-v3.5` is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial). Confirm that fits your deployment; commercial use requires permission from Jina.

## Local FAISS mode — in-memory reuse

When `USE_BEDROCK_KB=0`, the local chat model and FAISS index stay loaded in the **Python process** after the first load. Gradio page refreshes call `app.load` but **reuse** the same in-memory model and vector store instead of reloading from Hugging Face or re-extracting the embeddings zip.

## Arize Phoenix tracing (local)

Send retrieve/generate spans to a locally hosted [Arize Phoenix](https://docs.arize.com/phoenix) collector (same pattern as custom Pi spans in the doc-redaction agent — not LangChain auto-instrumentation).

1. Run Phoenix locally (UI typically at `http://localhost:6006`).
2. Set env vars (e.g. in `config/app_config.env`):

```
ARIZE_TRACING_ENABLED=1
ARIZE_BACKEND=phoenix
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=light-pdf-qa-chatbot
```

3. Install tracing deps (`arize-phoenix-otel`, OpenInference semconv, OpenTelemetry) from `requirements.txt`.
4. Ask a question in the Gradio UI; confirm `chat.retrieve` and `chat.generate` spans in Phoenix. Multi-turn chats share `session.id` = Gradio session hash.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ARIZE_TRACING_ENABLED` | `0` | Enable OTEL export |
| `ARIZE_BACKEND` | `phoenix` | `phoenix` (local) or `ax` (Arize cloud) |
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006` | Phoenix collector base URL |
| `PHOENIX_PROJECT_NAME` | `light-pdf-qa-chatbot` | Phoenix project name |
| `PHOENIX_API_KEY` | _(empty)_ | Optional if Phoenix requires auth |

For Arize AX (`ARIZE_BACKEND=ax`), set `ARIZE_SPACE_ID` / `ARIZE_API_KEY` and install `arize-otel`.
