import re
import os
import datetime
from typing import Type, Dict, List, Tuple
import time
from itertools import compress
import pandas as pd
import numpy as np
import google.generativeai as ai
from gradio import Progress
import boto3
import json

# Model packages
import torch.cuda
from threading import Thread
from transformers import pipeline, TextIteratorStreamer
from langchain_huggingface import HuggingFaceEmbeddings

# Alternative model sources
#from dataclasses import asdict, dataclass

# Langchain functions
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import SVMRetriever 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from chatfuncs.config import GEMINI_API_KEY, AWS_DEFAULT_REGION

model_object = [] # Define empty list for model functions to run
tokenizer = [] # Define empty list for model functions to run

from chatfuncs.model_load import temperature, max_new_tokens, sample, repetition_penalty, top_p, top_k, torch_device, CtransGenGenerationConfig, max_tokens

# ResponseObject class for AWS Bedrock calls
class ResponseObject:
    def __init__(self, text, usage_metadata):
        self.text = text
        self.usage_metadata = usage_metadata

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)

# For keyword extraction (not currently used)
#import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT

# For Name Entity Recognition model
#from span_marker import SpanMarkerModel # Not currently used

# For BM25 retrieval
import bm25s
import Stemmer

from chatfuncs.prompts import instruction_prompt_template_alpaca, instruction_prompt_mistral_orca, instruction_prompt_phi3, instruction_prompt_llama3, instruction_prompt_qwen, instruction_prompt_template_orca

import gradio as gr

torch.cuda.empty_cache()

PandasDataFrame = Type[pd.DataFrame]

embeddings = None  # global variable setup
vectorstore = None # global variable setup
model_type = None # global variable setup

max_memory_length = 0 # How long should the memory of the conversation last?

source_texts = "" # Define dummy source text (full text) just to enable highlight function to load


## Highlight text constants
hlt_chunk_size = 12
hlt_strat = [" ", ". ", "! ", "? ", ": ", "\n\n", "\n", ", "]
hlt_overlap = 4

## Initialise NER model ##
ner_model = []#SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd") # Not currently used

## Initialise keyword model ##
# Used to pull out keywords from chat history to add to user queries behind the scenes
kw_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

# Vectorstore funcs

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)
        
    '''  
    #with open("vectorstore.pkl", "wb") as f:
        #pickle.dump(vectorstore, f) 
    ''' 

    #if Path(save_to).exists():
    #    vectorstore_func.save_local(folder_path=save_to)
    #else:
    #    os.mkdir(save_to)
    #    vectorstore_func.save_local(folder_path=save_to)

    global vectorstore

    vectorstore = vectorstore_func

    out_message = "Document processing complete"

    #print(out_message)
    #print(f"> Saved to: {save_to}")

    return out_message

# Prompt functions

def base_prompt_templates(model_type:str = "Qwen 2 0.5B (small, fast)"):    
  
    #EXAMPLE_PROMPT = PromptTemplate(
    #    template="\nCONTENT:\n\n{page_content}\n\nSOURCE: {source}\n\n",
    #    input_variables=["page_content", "source"],
    #)

    CONTENT_PROMPT = PromptTemplate(
        template="{page_content}\n\n",#\n\nSOURCE: {source}\n\n",
        input_variables=["page_content"]
    )

# The main prompt:  

    if model_type == "Qwen 2 0.5B (small, fast)":
        INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_qwen, input_variables=['question', 'summaries'])
    elif model_type == "Phi 3.5 Mini (larger, slow)":
        INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_phi3, input_variables=['question', 'summaries'])
    else:
        INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_template_orca, input_variables=['question', 'summaries'])
        

    return INSTRUCTION_PROMPT, CONTENT_PROMPT

def write_out_metadata_as_string(metadata_in:str):
    metadata_string = [f"{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}" for d in metadata_in] # ['metadata']
    return metadata_string

def generate_expanded_prompt(inputs: Dict[str, str], instruction_prompt:str, content_prompt:str, extracted_memory:list, vectorstore:object, embeddings:object, relevant_flag:bool = True, out_passages:int = 2): # , 
        
    question =  inputs["question"]
    chat_history = inputs["chat_history"]
    
    if relevant_flag == True:
        new_question_kworded = adapt_q_from_chat_history(question, chat_history, extracted_memory) # new_question_keywords, 
        docs_keep_as_doc, doc_df, docs_keep_out = hybrid_retrieval(new_question_kworded, vectorstore, embeddings, k_val = 25, out_passages = out_passages, vec_score_cut_off = 0.85, vec_weight = 1, bm25_weight = 1, svm_weight = 1)
    else:
        new_question_kworded = question
        doc_df = pd.DataFrame()
        docs_keep_as_doc = []
        docs_keep_out = []
    
    if (not docs_keep_as_doc) | (doc_df.empty):
        sorry_prompt = """Say 'Sorry, there is no relevant information to answer this question.'"""
        return sorry_prompt, "No relevant sources found.", new_question_kworded
    
    # Expand the found passages to the neighbouring context
    if 'meta_url' in doc_df.columns:
        file_type = determine_file_type(doc_df['meta_url'][0])
    else:
        file_type = determine_file_type(doc_df['source'][0]) 

    # Only expand passages if not tabular data
    if (file_type != ".csv") & (file_type != ".xlsx"):
        docs_keep_as_doc, doc_df = get_expanded_passages(vectorstore, docs_keep_out, width=3)    

    # Build up sources content to add to user display
    doc_df['meta_clean'] = write_out_metadata_as_string(doc_df["metadata"]) # [f"<b>{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}</b>" for d in doc_df['metadata']]
    
    # Remove meta text from the page content if it already exists there
    doc_df['page_content_no_meta'] = doc_df.apply(lambda row: row['page_content'].replace(row['meta_clean'] + ". ", ""), axis=1)
    doc_df['content_meta'] = doc_df['meta_clean'].astype(str) + ".<br><br>" + doc_df['page_content_no_meta'].astype(str)

    #modified_page_content = [f" Document {i+1} - {word}" for i, word in enumerate(doc_df['page_content'])]
    modified_page_content = [f" Document {i+1} - {word}" for i, word in enumerate(doc_df['content_meta'])]
    docs_content_string = '<br><br>'.join(modified_page_content)

    sources_docs_content_string = '<br><br>'.join(doc_df['content_meta'])#.replace("  "," ")#.strip()
    
    instruction_prompt_out = instruction_prompt.format(question=new_question_kworded, summaries=docs_content_string)
    
    print('Final prompt is: ')
    print(instruction_prompt_out)
            
    return instruction_prompt_out, sources_docs_content_string, new_question_kworded

def create_full_prompt(user_input:str,
                       history:list[dict],
                       extracted_memory:str,
                       vectorstore:object,
                       embeddings:object,
                       model_type:str,
                       out_passages:list[str],
                       api_model_choice=None,
                       api_key=None,
                       relevant_flag = True):
    
    #if chain_agent is None:
    #    history.append((user_input, "Please click the button to submit the Huggingface API key before using the chatbot (top right)"))
    #    return history, history, "", ""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
        
    history = history or []

    if api_model_choice and api_model_choice != "None":
         print("API model choice detected")
         if api_key:
            print("API key detected")
            return history, "", None, relevant_flag       
         else:
            return history, "", None, relevant_flag
         
    # Create instruction prompt
    instruction_prompt, content_prompt = base_prompt_templates(model_type=model_type)

    if not user_input.strip():
        user_input = "No user input found"
        relevant_flag = False
    else:
        relevant_flag = True

    print("User input:", user_input)
   
    instruction_prompt_out, docs_content_string, new_question_kworded =\
                generate_expanded_prompt({"question": user_input, "chat_history": history}, #vectorstore,
                                    instruction_prompt, content_prompt, extracted_memory, vectorstore, embeddings, relevant_flag, out_passages)
  
    history.append({"metadata":None, "options":None, "role": 'user', "content": user_input})
    
    print("Output history is:", history)
    print("Final prompt to model is:",instruction_prompt_out)
        
    return history, docs_content_string, instruction_prompt_out, relevant_flag

def call_aws_claude(prompt: str, system_prompt: str, temperature: float, max_tokens: int, model_choice: str) -> ResponseObject:
    """
    This function sends a request to AWS Claude with the following parameters:
    - prompt: The user's input prompt to be processed by the model.
    - system_prompt: A system-defined prompt that provides context or instructions for the model.
    - temperature: A value that controls the randomness of the model's output, with higher values resulting in more diverse responses.
    - max_tokens: The maximum number of tokens (words or characters) in the model's response.
    - model_choice: The specific model to use for processing the request.
    
    The function constructs the request configuration, invokes the model, extracts the response text, and returns a ResponseObject containing the text and metadata.
    """

    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "top_p": 0.999,
        "temperature":temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }

    print("prompt_config:", prompt_config)

    body = json.dumps(prompt_config)

    modelId = model_choice
    accept = "application/json"
    contentType = "application/json"

    request = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    # Extract text from request
    response_body = json.loads(request.get("body").read())
    text = response_body.get("content")[0].get("text")

    response = ResponseObject(
    text=text,
    usage_metadata=request['ResponseMetadata']
    )

    # Now you can access both the text and metadata
    #print("Text:", response.text)
    print("Metadata:", response.usage_metadata)   
    
    return response

def construct_gemini_generative_model(in_api_key: str, temperature: float, model_choice: str, system_prompt: str, max_tokens: int) -> Tuple[object, dict]:
    """
    Constructs a GenerativeModel for Gemini API calls.

    Parameters:
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model, controlling the randomness of the output.
    - model_choice (str): The choice of model to use for generation.
    - system_prompt (str): The system prompt to guide the generation.
    - max_tokens (int): The maximum number of tokens to generate.

    Returns:
    - Tuple[object, dict]: A tuple containing the constructed GenerativeModel and its configuration.
    """
    # Construct a GenerativeModel
    try:
        if in_api_key:
            #print("Getting API key from textbox")
            api_key = in_api_key
            ai.configure(api_key=api_key)
        elif "GOOGLE_API_KEY" in os.environ:
            #print("Searching for API key in environmental variables")
            api_key = os.environ["GOOGLE_API_KEY"]
            ai.configure(api_key=api_key)
        else:
            print("No API key foound")
            raise gr.Error("No API key found.")
    except Exception as e:
        print(e)
    
    config = ai.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)

    print("model_choice:", model_choice)

    #model = ai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=config)
    model = ai.GenerativeModel(model_name=model_choice, system_instruction=system_prompt, generation_config=config)
    
    return model, config

# Function to send a request and update history
def send_request(prompt: str, conversation_history: List[dict], model: object, config: dict, model_choice: str, system_prompt: str, temperature: float, progress=Progress(track_tqdm=True)) -> Tuple[str, List[dict]]:
    """
    This function sends a request to a language model with the given prompt, conversation history, model configuration, model choice, system prompt, and temperature.
    It constructs the full prompt by appending the new user prompt to the conversation history, generates a response from the model, and updates the conversation history with the new prompt and response.
    If the model choice is specific to AWS Claude, it calls the `call_aws_claude` function; otherwise, it uses the `model.generate_content` method.
    The function returns the response text and the updated conversation history.
    """
    # Constructing the full prompt from the conversation history
    full_prompt = "Conversation history:\n"
    
    for entry in conversation_history:
        role = entry['role'].capitalize()  # Assuming the history is stored with 'role' and 'content'
        message = ' '.join(entry['parts'])  # Combining all parts of the message
        full_prompt += f"{role}: {message}\n"
    
    # Adding the new user prompt
    full_prompt += f"\nUser: {prompt}"

    # Print the full prompt for debugging purposes
    #print("full_prompt:", full_prompt)

    # Generate the model's response
    if "gemini" in model_choice:
        try:
            response = model.generate_content(contents=full_prompt, generation_config=config)
        except Exception as e:
            # If fails, try again after 10 seconds in case there is a throttle limit
            print(e)
            try:
                print("Calling Gemini model")
                out_message = "API limit hit - waiting 30 seconds to retry."
                print(out_message)
                progress(0.5, desc=out_message)
                time.sleep(30)
                response = model.generate_content(contents=full_prompt, generation_config=config)
            except Exception as e:
                print(e)
                return "", conversation_history
    elif "claude" in model_choice:
        try:
            print("Calling AWS Claude model")
            print("prompt:", prompt)
            print("system_prompt:", system_prompt)
            response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)
        except Exception as e:
            # If fails, try again after x seconds in case there is a throttle limit
            print(e)
            try:
                out_message = "API limit hit - waiting 30 seconds to retry."
                print(out_message)
                progress(0.5, desc=out_message)
                time.sleep(30)
                response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)
            
            except Exception as e:
                print(e)
                return "", conversation_history
    else:
        raise Exception("Model not found")

    # Update the conversation history with the new prompt and response
    conversation_history.append({"metadata":None, "options":None, "role": 'user', 'parts': [prompt]})
    conversation_history.append({"metadata":None, "options":None, "role": "assistant", 'parts': [response.text]})
    
    # Print the updated conversation history
    #print("conversation_history:", conversation_history)
    
    return response, conversation_history

def process_requests(prompts: List[str], system_prompt_with_table: str, conversation_history: List[dict], whole_conversation: List[str], whole_conversation_metadata: List[str], model: object, config: dict, model_choice: str, temperature: float, batch_no:int = 1, master:bool = False) -> Tuple[List[ResponseObject], List[dict], List[str], List[str]]:
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
    #for prompt in prompts:

    response, conversation_history = send_request(prompts[0], conversation_history, model=model, config=config, model_choice=model_choice, system_prompt=system_prompt_with_table, temperature=temperature)
    
    print(response.text)
    #"Okay, I'm ready. What source are we discussing, and what's your question about it? Please provide as much context as possible so I can give you the best answer."]
    print(response.usage_metadata)
    responses.append(response)

    # Create conversation txt object
    whole_conversation.append(prompts[0])
    whole_conversation.append(response.text)

    # Create conversation metadata
    if master == False:
        whole_conversation_metadata.append(f"Query batch {batch_no} prompt {len(responses)} metadata:")
    else:
        whole_conversation_metadata.append(f"Query summary metadata:")

    whole_conversation_metadata.append(str(response.usage_metadata))

    return responses, conversation_history, whole_conversation, whole_conversation_metadata

def produce_streaming_answer_chatbot(
            history:list,
            full_prompt:str,
            model_type:str,
            temperature:float=temperature,
            relevant_query_bool:bool=True,
            chat_history:list[dict]=[{"metadata":None, "options":None, "role": 'user', "content": ""}],
            max_new_tokens:int=max_new_tokens,
            sample:bool=sample,
            repetition_penalty:float=repetition_penalty,
            top_p:float=top_p,
            top_k:float=top_k,
            max_tokens:int=max_tokens,
            in_api_key:str=GEMINI_API_KEY
):
    #print("Model type is: ", model_type)

    #if not full_prompt.strip():
    #    if history is None:
    #        history = []

    #    return history

    history = chat_history

    print("history at start of streaming function:", history)

    if relevant_query_bool == False:
        history.append({"metadata":None, "options":None, "role": "assistant", "content": 'No relevant query found. Please retry your question'})

        yield history
        return

    if model_type == "Qwen 2 0.5B (small, fast)": 

        print("tokenizer:", tokenizer)
        # Get the model and tokenizer, and tokenize the user text.
        model_inputs = tokenizer(text=full_prompt, return_tensors="pt", return_attention_mask=False).to(torch_device)
        
        # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
        # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
        streamer = TextIteratorStreamer(tokenizer, timeout=120., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            temperature=temperature,
            top_k=top_k
        )

        print("model_object:", model_object)

        t = Thread(target=model_object.generate, kwargs=generate_kwargs)
        t.start()

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS=0
        print('-'*4+'Start Generation'+'-'*4)

        history.append({"metadata":None, "options":None, "role": "assistant", "content": ""})

        for new_text in streamer:
            try:
                if new_text is None:
                    new_text = ""
                history[-1]['content'] += new_text
                NUM_TOKENS += 1
                yield history
            except Exception as e:
                print(f"Error during text generation: {e}")
            
        time_generate = time.time() - start
        print('\n')
        print('-'*4+'End Generation'+'-'*4)
        print(f'Num of generated tokens: {NUM_TOKENS}')
        print(f'Time for complete generation: {time_generate}s')
        print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
        print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

    elif model_type == "Phi 3.5 Mini (larger, slow)":
        #tokens = model.tokenize(full_prompt)

        gen_config = CtransGenGenerationConfig()
        gen_config.update_temp(temperature)

        print(vars(gen_config))

        # Pull the generated text from the streamer, and update the model output.
        start = time.time()
        NUM_TOKENS=0
        print('-'*4+'Start Generation'+'-'*4)

        output = model_object(
        full_prompt, **vars(gen_config))

        history.append({"metadata":None, "options":None, "role": "assistant", "content": ""})

        for out in output:

            if "choices" in out and len(out["choices"]) > 0 and "text" in out["choices"][0]:
                history[-1]['content'] += out["choices"][0]["text"]
                NUM_TOKENS+=1
                yield history
            else:
                print(f"Unexpected output structure: {out}") 

        time_generate = time.time() - start
        print('\n')
        print('-'*4+'End Generation'+'-'*4)
        print(f'Num of generated tokens: {NUM_TOKENS}')
        print(f'Time for complete generation: {time_generate}s')
        print(f'Tokens per second: {NUM_TOKENS/time_generate}')
        print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

    elif "claude" in model_type:
        system_prompt = "You are answering questions from the user based on source material. Respond with short, factually correct answers."

        print("full_prompt:", full_prompt)

        if isinstance(full_prompt, str):
            full_prompt = [full_prompt]

        model = model_type
        config = {}

        responses, summary_conversation_history, whole_summary_conversation, whole_conversation_metadata = process_requests(full_prompt, system_prompt, conversation_history=[], whole_conversation=[], whole_conversation_metadata=[], model=model, config = config, model_choice = model_type, temperature = temperature)
        
        if isinstance(responses[-1], ResponseObject):
            response_texts = [resp.text for resp in responses]
        elif "choices" in responses[-1]:
            response_texts = [resp["choices"][0]['text'] for resp in responses]
        else:
            response_texts = [resp.text for resp in responses]

        latest_response_text = response_texts[-1]

        # Update the conversation history with the new prompt and response
        clean_text = re.sub(r'[\n\t\r]', ' ', latest_response_text)  # Replace newlines, tabs, and carriage returns with a space
        clean_response_text = re.sub(r'[^\x20-\x7E]', '', clean_text).strip()  # Remove all non-ASCII printable characters  
        
        history.append({"metadata":None, "options":None, "role": "assistant", "content": ''})
        
        for char in clean_response_text:
            time.sleep(0.005)
            history[-1]['content'] += char
            yield history
    
    elif "gemini" in model_type:
        print("Using Gemini model:", model_type)
        print("full_prompt:", full_prompt)

        if isinstance(full_prompt, str):
            full_prompt = [full_prompt]

        system_prompt = "You are answering questions from the user based on source material. Respond with short, factually correct answers."

        model, config = construct_gemini_generative_model(GEMINI_API_KEY, temperature, model_type, system_prompt, max_tokens)

        responses, summary_conversation_history, whole_summary_conversation, whole_conversation_metadata = process_requests(full_prompt, system_prompt, conversation_history=[], whole_conversation=[], whole_conversation_metadata=[], model=model, config = config, model_choice = model_type, temperature = temperature)

        if isinstance(responses[-1], ResponseObject):
            response_texts = [resp.text for resp in responses]
        elif "choices" in responses[-1]:
            response_texts = [resp["choices"][0]['text'] for resp in responses]
        else:
            response_texts = [resp.text for resp in responses]

        latest_response_text = response_texts[-1]

        clean_text = re.sub(r'[\n\t\r]', ' ', latest_response_text)  # Replace newlines, tabs, and carriage returns with a space
        clean_response_text = re.sub(r'[^\x20-\x7E]', '', clean_text).strip()  # Remove all non-ASCII printable characters   
        
        history.append({"metadata":None, "options":None, "role": "assistant", "content": ''})
        
        for char in clean_response_text:
            time.sleep(0.005)
            history[-1]['content'] += char
            yield history

        print("history at end of function:", history)

# Chat helper functions

def adapt_q_from_chat_history(question, chat_history, extracted_memory, keyword_model=""):#keyword_model): # new_question_keywords, 
 
        chat_history_str, chat_history_first_q, chat_history_first_ans, max_memory_length = _get_chat_history(chat_history)

        if chat_history_str:
            # Keyword extraction is now done in the add_inputs_to_history function
            #remove_q_stopwords(str(chat_history_first_q) + " " + str(chat_history_first_ans))
            
           
            new_question_kworded = str(extracted_memory) + ". " + question #+ " " + new_question_keywords
            #extracted_memory + " " + question
            
        else:
            new_question_kworded = question #new_question_keywords

        #print("Question output is: " + new_question_kworded)
            
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
            content=[]
            meta=[]
            meta_url=[]
            page_section=[]
            score=[]

            doc_df = pd.DataFrame()

            

            for item in docs_keep_out:
                content.append(item[0].page_content)
                meta.append(item[0].metadata)
                meta_url.append(item[0].metadata['source'])

                file_extension = determine_file_type(item[0].metadata['source'])
                if (file_extension != ".csv") & (file_extension != ".xlsx"):
                    page_section.append(item[0].metadata['page_section'])
                else: page_section.append("")
                score.append(item[1])       

            # Create df from 'winning' passages

            doc_df = pd.DataFrame(list(zip(content, meta, page_section, meta_url, score)),
               columns =['page_content', 'metadata', 'page_section', 'meta_url', 'score'])

            docs_content = doc_df['page_content'].astype(str)
            doc_df['full_url'] = "https://" + doc_df['meta_url'] 

            return doc_df

def hybrid_retrieval(new_question_kworded, vectorstore, embeddings, k_val, out_passages,
                           vec_score_cut_off, vec_weight, bm25_weight, svm_weight): # ,vectorstore, embeddings

            #vectorstore=globals()["vectorstore"]
            #embeddings=globals()["embeddings"]
            doc_df = pd.DataFrame()


            docs = vectorstore.similarity_search_with_score(new_question_kworded, k=k_val)

            # Keep only documents with a certain score
            docs_len = [len(x[0].page_content) for x in docs]
            docs_scores = [x[1] for x in docs]

            # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
            score_more_limit = pd.Series(docs_scores) < vec_score_cut_off
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

                content=[]
                meta_url=[]
                score=[]
                
                for item in docs_keep:
                    content.append(item[0].page_content)
                    meta_url.append(item[0].metadata['source'])
                    score.append(item[1])       

                # Create df from 'winning' passages

                doc_df = pd.DataFrame(list(zip(content, meta_url, score)),
                columns =['page_content', 'meta_url', 'score'])

                docs_content = doc_df['page_content'].astype(str)
                docs_url = doc_df['meta_url']

                return docs_keep_as_doc, doc_df, docs_content, docs_url
            
            # Check for if more docs are removed than the desired output
            if out_passages > docs_keep_length: 
                out_passages = docs_keep_length
                k_val = docs_keep_length
                     
            vec_rank = [*range(1, docs_keep_length+1)]
            vec_score = [(docs_keep_length/x)*vec_weight for x in vec_rank]

            print("Number of documents remaining: ", docs_keep_length)
            
            # 2nd level check using BM25s package to do keyword search on retrieved passages.
            
            content_keep=[]
            for item in docs_keep:
                content_keep.append(item[0].page_content)

            # Prepare Corpus (Tokenized & Optional Stemming)
            corpus = [doc.lower() for doc in content_keep]
            #stemmer = SnowballStemmer("english", ignore_stopwords=True)  # NLTK stemming not compatible
            stemmer = Stemmer.Stemmer("english")
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

            # Create and Index with BM25s
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

            # Query Processing (Stemming applied consistently if used above)
            query_tokens = bm25s.tokenize(new_question_kworded.lower(), stemmer=stemmer)
            results, scores = retriever.retrieve(query_tokens, corpus=corpus, k=len(corpus)) # Retrieve all docs

            for i in range(results.shape[1]):
                doc, score = results[0, i], scores[0, i]
                print(f"Rank {i+1} (score: {score:.2f}): {doc}")

            #print("BM25 results:", results)
            #print("BM25 scores:", scores)

            # Rank Calculation (Custom Logic for Your BM25 Score)
            bm25_rank = list(range(1, len(results[0]) + 1))
            #bm25_rank = results[0]#.tolist()[0]  # Since you have a single query
            bm25_score = [(docs_keep_length / (rank + 1)) * bm25_weight for rank in bm25_rank] 
            # +1 to avoid division by 0 for rank 0

            # Result Ordering (Using the calculated ranks)
            pairs = list(zip(bm25_rank, docs_keep_as_doc))
            pairs.sort()
            bm25_result = [value for rank, value in pairs]
            

            # 3rd level check on retrieved docs with SVM retriever
            # Check the type of the embeddings object
            embeddings_type = type(embeddings)


            #hf_embeddings = HuggingFaceEmbeddings(**embeddings)
            hf_embeddings = embeddings
            
            svm_retriever = SVMRetriever.from_texts(content_keep, hf_embeddings, k = k_val)
            svm_result = svm_retriever.invoke(new_question_kworded)

         
            svm_rank=[]
            svm_score = []

            for vec_item in docs_keep:
                x = 0
                for svm_item in svm_result:
                    x = x + 1
                    if svm_item.page_content == vec_item[0].page_content:
                        svm_rank.append(x)
                        svm_score.append((docs_keep_length/x)*svm_weight)

        
            ## Calculate final score based on three ranking methods
            final_score = [a  + b + c for a, b, c in zip(vec_score, bm25_score, svm_score)]
            final_rank = [sorted(final_score, reverse=True).index(x)+1 for x in final_score]
            # Force final_rank to increment by 1 each time
            final_rank = list(pd.Series(final_rank).rank(method='first'))

            #print("final rank: " + str(final_rank))
            #print("out_passages: " + str(out_passages))

            best_rank_index_pos = []

            for x in range(1,out_passages+1):
                try:
                    best_rank_index_pos.append(final_rank.index(x))
                except IndexError: # catch the error
                    pass

            # Adjust best_rank_index_pos to 

            best_rank_pos_series = pd.Series(best_rank_index_pos)


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
        return ''.join(content), meta[0], meta[-1]
    
    def get_parent_content_and_meta(vstore_docs, width, target):
        #target_range = range(max(0, target - width), min(len(vstore_docs), target + width + 1))
        target_range = range(max(0, target), min(len(vstore_docs), target + width + 1)) # Now only selects extra passages AFTER the found passage
        parent_vstore_out = [vstore_docs[i] for i in target_range]
        
        content_str_out, meta_first_out, meta_last_out = [], [], []
        for _ in parent_vstore_out:
            content_str, meta_first, meta_last = extract_details(parent_vstore_out)
            content_str_out.append(content_str)
            meta_first_out.append(meta_first)
            meta_last_out.append(meta_last)
        return content_str_out, meta_first_out, meta_last_out

    def merge_dicts_except_source(d1, d2):
            merged = {}
            for key in d1:
                if key != "source":
                    merged[key] = str(d1[key]) + " to " + str(d2[key])
                else:
                    merged[key] = d1[key]  # or d2[key], based on preference
            return merged

    def merge_two_lists_of_dicts(list1, list2):
        return [merge_dicts_except_source(d1, d2) for d1, d2 in zip(list1, list2)]

    # Step 1: Filter vstore_docs
    vstore_docs = get_docs_from_vstore(vectorstore)
    doc_sources = {doc.metadata['source'] for doc, _ in docs}
    vstore_docs = [(k, v) for k, v in vstore_docs if v.metadata.get('source') in doc_sources]

    # Step 2: Group by source and proceed
    vstore_by_source = defaultdict(list)
    for k, v in vstore_docs:
        vstore_by_source[v.metadata['source']].append((k, v))
        
    expanded_docs = []
    for doc, score in docs:
        search_source = doc.metadata['source']
        

        #if file_type == ".csv" | file_type == ".xlsx":
        #     content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], 0, search_index)

        #else:
        search_section = doc.metadata['page_section']
        parent_vstore_meta_section = [doc.metadata['page_section'] for _, doc in vstore_by_source[search_source]]
        search_index = parent_vstore_meta_section.index(search_section) if search_section in parent_vstore_meta_section else -1

        content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_by_source[search_source], width, search_index)
        meta_full = merge_two_lists_of_dicts(meta_first, meta_last)

        expanded_doc = (Document(page_content=content_str[0], metadata=meta_full[0]), score)
        expanded_docs.append(expanded_doc)

    doc_df = pd.DataFrame()

    doc_df = create_doc_df(expanded_docs)  # Assuming you've defined the 'create_doc_df' function elsewhere

    return expanded_docs, doc_df

def highlight_found_text(chat_history: list[dict], source_texts: list[dict], hlt_chunk_size:int=hlt_chunk_size, hlt_strat:List=hlt_strat, hlt_overlap:int=hlt_overlap) -> str:
    """
    Highlights occurrences of chat_history within source_texts.
    
    Parameters:
    - chat_history (str): The text to be searched for within source_texts.
    - source_texts (str): The text within which chat_history occurrences will be highlighted.
    
    Returns:
    - str: A string with occurrences of chat_history highlighted.
    
    Example:
    >>> highlight_found_text("world", "Hello, world! This is a test. Another world awaits.")
    'Hello, <mark style="color:black;">world</mark>! This is a test. Another <mark style="color:black;">world</mark> awaits.'
    """

    def extract_text_from_input(text, i=0):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[i][0].replace("  ", " ").strip()
        else:
            return ""
        
    print("chat_history:", chat_history)
        
    response_text = next(
    (entry['content'] for entry in reversed(chat_history) if entry.get('role') == 'assistant'),
    "")

    print("response_text:", response_text)
        
    source_texts = extract_text_from_input(source_texts)

    print("source_texts:", source_texts)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hlt_chunk_size,
        separators=hlt_strat,
        chunk_overlap=hlt_overlap,
    )
    sections = text_splitter.split_text(response_text)

    found_positions = {}
    for x in sections:
        text_start_pos = 0
        while text_start_pos != -1:
            text_start_pos = source_texts.find(x, text_start_pos)
            if text_start_pos != -1:
                found_positions[text_start_pos] = text_start_pos + len(x)
                text_start_pos += 1

    # Combine overlapping or adjacent positions
    sorted_starts = sorted(found_positions.keys())
    combined_positions = []
    if sorted_starts:
        current_start, current_end = sorted_starts[0], found_positions[sorted_starts[0]]
        for start in sorted_starts[1:]:
            if start <= (current_end + 10):
                current_end = max(current_end, found_positions[start])
            else:
                combined_positions.append((current_start, current_end))
                current_start, current_end = start, found_positions[start]
        combined_positions.append((current_start, current_end))

    # Construct pos_tokens
    pos_tokens = []
    prev_end = 0
    for start, end in combined_positions:
        if end-start > 15: # Only combine if there is a significant amount of matched text. Avoids picking up single words like 'and' etc.
            pos_tokens.append(source_texts[prev_end:start])
            pos_tokens.append('<mark style="color:black;">' + source_texts[start:end] + '</mark>')
            prev_end = end
    pos_tokens.append(source_texts[prev_end:])

    out_pos_tokens = "".join(pos_tokens)

    print("out_pos_tokens:", out_pos_tokens)

    return out_pos_tokens


# # Chat history functions

def clear_chat(chat_history_state, sources, chat_message, current_topic):
    chat_history_state = None
    sources = ''
    chat_message = None
    current_topic = ''

    return chat_history_state, sources, chat_message, current_topic

def _get_chat_history(chat_history: List[Tuple[str, str]], max_memory_length:int = max_memory_length): # Limit to last x interactions only

    if (not chat_history) | (max_memory_length == 0):
        chat_history = []

    if len(chat_history) > max_memory_length:
        chat_history = chat_history[-max_memory_length:]
        
    #print(chat_history)

    first_q = ""
    first_ans = ""
    for human_s, ai_s in chat_history:
        first_q = human_s
        first_ans = ai_s

        #print("Text to keyword extract: " + first_q + " " + first_ans)
        break

    conversation = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        conversation += "\n" + "\n".join([human, ai])

    return conversation, first_q, first_ans, max_memory_length

def add_inputs_answer_to_history(user_message, history, current_topic):
    
    if history is None:
        history = [("","")]

    #history.append((user_message, [-1]))

    chat_history_str, chat_history_first_q, chat_history_first_ans, max_memory_length = _get_chat_history(history)


    # Only get the keywords for the first question and response, or do it every time if over 'max_memory_length' responses in the conversation
    if (len(history) == 1) | (len(history) > max_memory_length):
        
        #print("History after appending is:")
        #print(history)

        first_q_and_first_ans = str(chat_history_first_q) + " " + str(chat_history_first_ans)
        #ner_memory = remove_q_ner_extractor(first_q_and_first_ans)
        keywords = keybert_keywords(first_q_and_first_ans, n = 8, kw_model=kw_model)
        #keywords.append(ner_memory)

        # Remove duplicate words while preserving order
        ordered_tokens = set()
        result = []
        for word in keywords:
                if word not in ordered_tokens:
                        ordered_tokens.add(word)
                        result.append(word)

        extracted_memory = ' '.join(result)

    else: extracted_memory=current_topic
    
    print("Extracted memory is:")
    print(extracted_memory)
    
    
    return history, extracted_memory

# Keyword functions

def remove_q_stopwords(question): # Remove stopwords from question. Not used at the moment 
    # Prepare keywords from question by removing stopwords
    text = question.lower()

    # Remove numbers
    text = re.sub('[0-9]', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    text_tokens = tokenizer.tokenize(text)
    #text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in tokens_without_sw:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)   


    new_question_keywords = ' '.join(result)
    return new_question_keywords

def remove_q_ner_extractor(question):
    
    predict_out = ner_model.predict(question)
    predict_tokens = [' '.join(v for k, v in d.items() if k == 'span') for d in predict_out]

    # Remove duplicate words while preserving order
    ordered_tokens = set()
    result = []
    for word in predict_tokens:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
     


    new_question_keywords = ' '.join(result).lower()
    return new_question_keywords

def apply_lemmatize(text, wnl=WordNetLemmatizer()):

    def prep_for_lemma(text):

        # Remove numbers
        text = re.sub('[0-9]', '', text)
        print(text)

        tokenizer = RegexpTokenizer(r'\w+')
        text_tokens = tokenizer.tokenize(text)
        #text_tokens = word_tokenize(text)

        return text_tokens

    tokens = prep_for_lemma(text)

    def lem_word(word):
    
        if len(word) > 3: out_word = wnl.lemmatize(word)
        else: out_word = word

        return out_word

    return [lem_word(token) for token in tokens]

def keybert_keywords(text, n, kw_model):
    tokens_lemma = apply_lemmatize(text)
    lemmatised_text = ' '.join(tokens_lemma)

    keywords_text = KeyBERT(model=kw_model).extract_keywords(lemmatised_text, stop_words='english', top_n=n, 
                                                   keyphrase_ngram_range=(1, 1))
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

def vote(data: gr.LikeData, chat_history, instruction_prompt_out, model_type):
    import os
    import pandas as pd

    chat_history_last = str(str(chat_history[-1][0]) + " - " + str(chat_history[-1][1]))

    response_df = pd.DataFrame(data={"thumbs_up":data.liked,
                                        "chosen_response":data.value,
                                          "input_prompt":instruction_prompt_out,
                                          "chat_history":chat_history_last,
                                          "model_type": model_type,
                                          "date_time": pd.Timestamp.now()}, index=[0])

    if data.liked:
        print("You upvoted this response: " + data.value)
        
        if os.path.isfile("thumbs_up_data.csv"):
             existing_thumbs_up_df = pd.read_csv("thumbs_up_data.csv")
             thumbs_up_df_concat = pd.concat([existing_thumbs_up_df, response_df], ignore_index=True).drop("Unnamed: 0",axis=1, errors="ignore")
             thumbs_up_df_concat.to_csv("thumbs_up_data.csv")
        else:
            response_df.to_csv("thumbs_up_data.csv")

    else:
        print("You downvoted this response: " + data.value)

        if os.path.isfile("thumbs_down_data.csv"):
             existing_thumbs_down_df = pd.read_csv("thumbs_down_data.csv")
             thumbs_down_df_concat = pd.concat([existing_thumbs_down_df, response_df], ignore_index=True).drop("Unnamed: 0",axis=1, errors="ignore")
             thumbs_down_df_concat.to_csv("thumbs_down_data.csv")
        else:
            response_df.to_csv("thumbs_down_data.csv")            
