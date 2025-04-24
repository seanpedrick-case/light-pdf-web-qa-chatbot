import os
from typing import Type
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
import pandas as pd
from torch import float16
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,  AutoModelForCausalLM
import zipfile

from chatfuncs.ingest import embed_faiss_save_to_zip

from chatfuncs.helper_functions import get_connection_params, reveal_feedback_buttons, wipe_logs
from chatfuncs.aws_functions import upload_file_to_s3
from chatfuncs.auth import authenticate_user
from chatfuncs.config import FEEDBACK_LOGS_FOLDER, ACCESS_LOGS_FOLDER, USAGE_LOGS_FOLDER, HOST_NAME, COGNITO_AUTH, INPUT_FOLDER, OUTPUT_FOLDER, MAX_QUEUE_SIZE, DEFAULT_CONCURRENCY_LIMIT, MAX_FILE_SIZE, GRADIO_SERVER_PORT, ROOT_PATH, DEFAULT_EMBEDDINGS_LOCATION, EMBEDDINGS_MODEL_NAME, DEFAULT_DATA_SOURCE, HF_TOKEN, LARGE_MODEL_REPO_ID, LARGE_MODEL_GGUF_FILE, LARGE_MODEL_NAME, SMALL_MODEL_NAME, SMALL_MODEL_REPO_ID, DEFAULT_DATA_SOURCE_NAME, DEFAULT_EXAMPLES, DEFAULT_MODEL_CHOICES
from chatfuncs.model_load import torch_device, gpu_config, cpu_config, context_length
import chatfuncs.chatfuncs as chatf
import chatfuncs.ingest as ing

PandasDataFrame = Type[pd.DataFrame]

from datetime import datetime
today_rev = datetime.now().strftime("%Y%m%d")

host_name = HOST_NAME
access_logs_data_folder = ACCESS_LOGS_FOLDER
feedback_data_folder = FEEDBACK_LOGS_FOLDER
usage_data_folder = USAGE_LOGS_FOLDER

if isinstance(DEFAULT_EXAMPLES, str): default_examples_set = eval(DEFAULT_EXAMPLES)
if isinstance(DEFAULT_MODEL_CHOICES, str): default_model_choices = eval(DEFAULT_MODEL_CHOICES)

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 


###
# Load preset embeddings, vectorstore, and model
###

def load_embeddings_model(embeddings_model = EMBEDDINGS_MODEL_NAME):

    embeddings_func = HuggingFaceEmbeddings(model_name=embeddings_model)

    #global embeddings

    #embeddings = embeddings_func

    return embeddings_func

def get_faiss_store(faiss_vstore_folder:str, embeddings_model:object):

    with zipfile.ZipFile(faiss_vstore_folder + '/' + faiss_vstore_folder + '.zip', 'r') as zip_ref:
        zip_ref.extractall(faiss_vstore_folder)

    faiss_vstore = FAISS.load_local(folder_path=faiss_vstore_folder, embeddings=embeddings_model, allow_dangerous_deserialization=True)
    os.remove(faiss_vstore_folder + "/index.faiss")
    os.remove(faiss_vstore_folder + "/index.pkl")
    
    #global vectorstore

    #vectorstore = faiss_vstore

    return faiss_vstore #vectorstore

# Load in default embeddings and embeddings model name
embeddings_model = load_embeddings_model(EMBEDDINGS_MODEL_NAME)
vectorstore = get_faiss_store(faiss_vstore_folder=DEFAULT_EMBEDDINGS_LOCATION,embeddings_model=embeddings_model)#globals()["embeddings"])

chatf.embeddings = embeddings_model
chatf.vectorstore = vectorstore

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings_model=embeddings_model):

    print(f"> Total split documents: {len(docs_out)}")

    print(docs_out)

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings_model)

    chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, vectorstore_func
 

def create_hf_model(model_name:str, hf_token=HF_TOKEN):
    if torch_device == "cuda":
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")#, torch_dtype=torch.float16)
        else:
            if hf_token:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=hf_token) # , torch_dtype=float16
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto") # , torch_dtype=float16
    else:
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.float16)
        else:
            if hf_token:
                model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token) # , torch_dtype=float16
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name) # , torch_dtype=float16

    if hf_token:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = context_length, token=hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = context_length)

    return model, tokenizer

def load_model(model_type:str, gpu_layers:int, gpu_config:dict=gpu_config, cpu_config:dict=cpu_config, torch_device:str=torch_device):
    print("Loading model")

    if model_type == LARGE_MODEL_NAME:
        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
            print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU.")
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

            print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU.")

        try:
            model = Llama(
            model_path=hf_hub_download(
            repo_id=LARGE_MODEL_REPO_ID,
            filename=LARGE_MODEL_GGUF_FILE 
        ),
        **vars(gpu_config) # change n_gpu_layers if you have more or less VRAM 
        )
        
        except Exception as e:
            print("GPU load failed", e, "loading CPU version instead")
            model = Llama(
            model_path=hf_hub_download(
            repo_id=LARGE_MODEL_REPO_ID,
            filename=LARGE_MODEL_GGUF_FILE
        ),
        **vars(cpu_config)
        )

        tokenizer = []

    if model_type == SMALL_MODEL_NAME:
        # Huggingface chat model
        hf_checkpoint = SMALL_MODEL_REPO_ID# 'declare-lab/flan-alpaca-large'#'declare-lab/flan-alpaca-base' # # # 'Qwen/Qwen1.5-0.5B-Chat' #
        
        model, tokenizer = create_hf_model(model_name = hf_checkpoint)

    else:
        model = model_type
        tokenizer = ""

    chatf.model_object = model
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)

    return model_type, load_confirmation, model_type#model, tokenizer, model_type


###
# RUN UI
###

app = gr.Blocks(theme = gr.themes.Base(), fill_width=True)#css=".gradio-container {background-color: black}")

with app:
    model_type = SMALL_MODEL_NAME
    load_model(model_type, 0, gpu_config, cpu_config, torch_device) # chatf.model_object, chatf.tokenizer, chatf.model_type = 

    # Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
    #model_type = "Phi 3.5 Mini (larger, slow)"
    #load_model(model_type, gpu_layers, gpu_config, cpu_config, torch_device)

    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()

    model_type_state = gr.State(model_type)
    gpu_config_state = gr.State(gpu_config)
    cpu_config_state = gr.State(cpu_config)
    torch_device_state = gr.State(torch_device)

    # Embeddings related vars
    embeddings_model_object_state = gr.State(embeddings_model)#globals()["embeddings"])
    vectorstore_state = gr.State(vectorstore)#globals()["vectorstore"]) 
    default_embeddings_store_text = gr.Textbox(value=DEFAULT_EMBEDDINGS_LOCATION, visible=False)

    # Is the query relevant to the sources provided?
    relevant_query_state = gr.Checkbox(value=True, visible=False) 

    # Storing model objects in state doesn't seem to work, so we have to load in different models in roundabout ways
    model_state = gr.State() # chatf.model_object (gives error)
    tokenizer_state = gr.State() # chatf.tokenizer (gives error)

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    session_hash_state = gr.State()
    output_folder_textbox = gr.Textbox(value=OUTPUT_FOLDER, visible=False)
    input_folder_textbox = gr.Textbox(value=INPUT_FOLDER, visible=False)

    session_hash_textbox = gr.Textbox(value="", visible=False)
    s3_logs_output_textbox = gr.Textbox(label="S3 logs", visible=False)

    access_logs_state = gr.State(access_logs_data_folder + 'dataset1.csv')
    access_s3_logs_loc_state = gr.State(access_logs_data_folder)
    usage_logs_state = gr.State(usage_data_folder + 'dataset1.csv')
    usage_s3_logs_loc_state = gr.State(usage_data_folder)
    feedback_logs_state = gr.State(feedback_data_folder + 'dataset1.csv')
    feedback_s3_logs_loc_state = gr.State(feedback_data_folder)

    gr.Markdown("<h1><center>Lightweight PDF / web page QA bot</center></h1>")        
    
    gr.Markdown(f"""Chat with PDF, web page or (new) csv/Excel documents. The default is a small model ({SMALL_MODEL_NAME}), that can only answer specific questions that are answered in the text. It cannot give overall impressions of, or summarise the document. The alternative ({LARGE_MODEL_NAME}), can reason a little better, but is much slower (See Advanced settings tab).\n\nBy default '[{DEFAULT_DATA_SOURCE_NAME}]({DEFAULT_DATA_SOURCE})' is loaded.If you want to talk about another document or web page, please select from the second tab. If switching topic, please click the 'Clear chat' button.\n\nCaution: This is a public app. Please ensure that the document you upload is not sensitive is any way as other users may see it! Also, please note that LLM chatbots may give incomplete or incorrect information, so please use with care.""")
        

    with gr.Row():
        current_source = gr.Textbox(label="Current data source(s)", value=DEFAULT_DATA_SOURCE, scale = 10)
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3)

    with gr.Tab("Chatbot"):

        with gr.Row():
            #chat_height = 500
            chatbot = gr.Chatbot(value=None, avatar_images=('user.jfif', 'bot.jpg'), scale = 1, resizable=True, show_copy_all_button=True, show_copy_button=True, show_share_button=True, type='messages') # , height=chat_height
            with gr.Accordion("Open this tab to see the source paragraphs used to generate the answer", open = True):
                sources = gr.HTML(value = "Source paragraphs with the most relevant text will appear here") # , height=chat_height

        with gr.Row():
            message = gr.Textbox(
                label="Enter your question here",
                lines=1,
            )     
        with gr.Row():
            submit = gr.Button(value="Send message", variant="primary", scale = 4)
            clear = gr.Button(value="Clear chat", variant="secondary", scale=1) 
            stop = gr.Button(value="Stop generating", variant="stop", scale=1)

        examples_set = gr.Radio(label="Example questions",
            choices=default_examples_set)
        
        current_topic = gr.Textbox(label="Feature currently disabled - Keywords related to current conversation topic.", placeholder="Keywords related to the conversation topic will appear here", visible=False)

    with gr.Tab("Load in a different file to chat with"):
        with gr.Accordion("PDF file", open = False):
            in_pdf = gr.File(label="Upload pdf", file_count="multiple", file_types=['.pdf'])
            load_pdf = gr.Button(value="Load in file", variant="secondary", scale=0)
        
        with gr.Accordion("Web page", open = False):
            with gr.Row():
                in_web = gr.Textbox(label="Enter web page url")
                in_div = gr.Textbox(label="(Advanced) Web page div for text extraction", value="p", placeholder="p")
            load_web = gr.Button(value="Load in webpage", variant="secondary", scale=0)

        with gr.Accordion("CSV/Excel file", open = False):
            in_csv = gr.File(label="Upload CSV/Excel file", file_count="multiple", file_types=['.csv', '.xlsx'])
            in_text_column = gr.Textbox(label="Enter column name where text is stored")
            load_csv = gr.Button(value="Load in CSV/Excel file", variant="secondary", scale=0)

        with gr.Row():
            ingest_embed_out = gr.Textbox(label="File/web page preparation progress")
            file_out_box = gr.File(file_count='single', file_types=['.zip'])

    with gr.Tab("Advanced features"):
        out_passages = gr.Slider(minimum=1, value = 2, maximum=10, step=1, label="Choose number of passages to retrieve from the document. Numbers greater than 2 may lead to increased hallucinations or input text being truncated.")
        temp_slide = gr.Slider(minimum=0.1, value = 0.5, maximum=1, step=0.1, label="Choose temperature setting for response generation.")
        with gr.Row():
            model_choice = gr.Radio(label="Choose a chat model", value=SMALL_MODEL_NAME, choices = default_model_choices)
            in_api_key = gr.Textbox(value = "", label="Enter Gemini API key (only if using Google API models)", lines=1, type="password",interactive=True, visible=True)
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=100, step = 1, visible=True)
            
        load_text = gr.Text(label="Load status")        

    gr.HTML(
        "<center>This app is powered by Gradio, Transformers, and Llama.cpp.</center>"
    )

    examples_set.change(fn=chatf.update_message, inputs=[examples_set], outputs=[message])

    ###
    # CHAT PAGE
    ###

    # Click to send message
    response_click = submit.click(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_model_object_state, model_type_state, out_passages, in_api_key], outputs=[chat_history_state, sources, instruction_prompt_out, relevant_query_state], queue=False, api_name="retrieval").\
                success(chatf.turn_off_interactivity, inputs=None, outputs=[message, submit], queue=False).\
                success(chatf.produce_streaming_answer_chatbot, inputs=[chatbot, instruction_prompt_out, model_type_state, temp_slide, relevant_query_state, chat_history_state, in_api_key], outputs=chatbot)
    response_click.success(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                success(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                success(lambda: chatf.restore_interactivity(), None, [message, submit], queue=False)

    # Press enter to send message
    response_enter = message.submit(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_model_object_state, model_type_state, out_passages, in_api_key], outputs=[chat_history_state, sources, instruction_prompt_out, relevant_query_state], queue=False).\
                success(chatf.turn_off_interactivity, inputs=None, outputs=[message, submit], queue=False).\
                success(chatf.produce_streaming_answer_chatbot, [chatbot, instruction_prompt_out, model_type_state, temp_slide, relevant_query_state, chat_history_state, in_api_key], chatbot)
    response_enter.success(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                success(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                success(lambda: chatf.restore_interactivity(), None, [message, submit], queue=False)
    
    # Stop box
    stop.click(fn=None, inputs=None, outputs=None, cancels=[response_click, response_enter])

    # Clear box
    clear.click(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic])
    clear.click(lambda: None, None, chatbot, queue=False)

    # Thumbs up or thumbs down voting function
    chatbot.like(chatf.vote, [chat_history_state, instruction_prompt_out, model_type_state], None)
    

    ###
    # LOAD NEW DATA PAGE
    ###

    # Load in a pdf
    load_pdf_click = load_pdf.click(ing.parse_file, inputs=[in_pdf], outputs=[ingest_text, current_source]).\
             success(ing.text_to_docs, inputs=[ingest_text], outputs=[ingest_docs]).\
             success(embed_faiss_save_to_zip, inputs=[ingest_docs, output_folder_textbox, embeddings_model_object_state], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             success(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage
    load_web_click = load_web.click(ing.parse_html, inputs=[in_web, in_div], outputs=[ingest_text, ingest_metadata, current_source]).\
             success(ing.html_text_to_docs, inputs=[ingest_text, ingest_metadata], outputs=[ingest_docs]).\
             success(embed_faiss_save_to_zip, inputs=[ingest_docs, output_folder_textbox, embeddings_model_object_state], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             success(chatf.hide_block, outputs = [examples_set])
    
    # Load in a csv/excel file
    load_csv_click = load_csv.click(ing.parse_csv_or_excel, inputs=[in_csv, in_text_column], outputs=[ingest_text, current_source]).\
             success(ing.csv_excel_text_to_docs, inputs=[ingest_text, in_text_column], outputs=[ingest_docs]).\
             success(embed_faiss_save_to_zip, inputs=[ingest_docs, output_folder_textbox, embeddings_model_object_state], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             success(chatf.hide_block, outputs = [examples_set])
   

    ###
    # LOAD MODEL PAGE
    ###

    change_model_button.click(fn=chatf.turn_off_interactivity, inputs=None, outputs=[message, submit], queue=False).\
    success(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state, load_text, current_model]).\
    success(lambda: chatf.restore_interactivity(), None, [message, submit], queue=False).\
    success(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic]).\
    success(lambda: None, None, chatbot, queue=False)

    ###
    # LOGGING AND ON APP LOAD FUNCTIONS
    ###  
    # Load in default model and embeddings for each user  
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, output_folder_textbox, session_hash_textbox, input_folder_textbox]).\
    success(load_model, inputs=[model_type_state, gpu_layer_choice, gpu_config_state, cpu_config_state, torch_device_state], outputs=[model_type_state, load_text, current_model]).\
    success(get_faiss_store, inputs=[default_embeddings_store_text, embeddings_model_object_state], outputs=[vectorstore_state])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger()
    access_callback.setup([session_hash_textbox], access_logs_data_folder)

    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    success(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

if __name__ == "__main__":
    if COGNITO_AUTH == "1":
        app.queue(max_size=int(MAX_QUEUE_SIZE), default_concurrency_limit=int(DEFAULT_CONCURRENCY_LIMIT)).launch(show_error=True, inbrowser=True, auth=authenticate_user, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)
    else:
        app.queue(max_size=int(MAX_QUEUE_SIZE), default_concurrency_limit=int(DEFAULT_CONCURRENCY_LIMIT)).launch(show_error=True, inbrowser=True, max_file_size=MAX_FILE_SIZE, server_port=GRADIO_SERVER_PORT, root_path=ROOT_PATH)