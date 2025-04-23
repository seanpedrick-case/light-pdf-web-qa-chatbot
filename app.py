import os
from typing import Type
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
import pandas as pd

from chatfuncs.ingest import embed_faiss_save_to_zip

from chatfuncs.helper_functions import ensure_output_folder_exists, get_connection_params, output_folder, reveal_feedback_buttons, wipe_logs
from chatfuncs.aws_functions import upload_file_to_s3
from chatfuncs.auth import authenticate_user
from chatfuncs.config import FEEDBACK_LOGS_FOLDER, ACCESS_LOGS_FOLDER, USAGE_LOGS_FOLDER, HOST_NAME, COGNITO_AUTH

from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,  AutoModelForCausalLM
import os

PandasDataFrame = Type[pd.DataFrame]

from datetime import datetime
today_rev = datetime.now().strftime("%Y%m%d")

host_name = HOST_NAME
access_logs_data_folder = ACCESS_LOGS_FOLDER
feedback_data_folder = FEEDBACK_LOGS_FOLDER
usage_data_folder = USAGE_LOGS_FOLDER

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import chatfuncs.ingest as ing

###
# Load preset embeddings, vectorstore, and model
###

embeddings_name =  "BAAI/bge-base-en-v1.5" #"mixedbread-ai/mxbai-embed-xsmall-v1" 

def load_embeddings(embeddings_name = embeddings_name):

    embeddings_func = HuggingFaceEmbeddings(model_name=embeddings_name)

    global embeddings

    embeddings = embeddings_func

    return embeddings

def get_faiss_store(faiss_vstore_folder,embeddings):
    import zipfile
    with zipfile.ZipFile(faiss_vstore_folder + '/' + faiss_vstore_folder + '.zip', 'r') as zip_ref:
        zip_ref.extractall(faiss_vstore_folder)

    faiss_vstore = FAISS.load_local(folder_path=faiss_vstore_folder, embeddings=embeddings, allow_dangerous_deserialization=True)
    os.remove(faiss_vstore_folder + "/index.faiss")
    os.remove(faiss_vstore_folder + "/index.pkl")
    
    global vectorstore

    vectorstore = faiss_vstore

    return vectorstore

import chatfuncs.chatfuncs as chatf
from chatfuncs.model_load import torch_device, gpu_config, cpu_config, context_length

chatf.embeddings = load_embeddings(embeddings_name)
chatf.vectorstore = get_faiss_store(faiss_vstore_folder="faiss_embedding",embeddings=globals()["embeddings"])

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    print(docs_out)

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)

    chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, vectorstore_func
 # Gradio chat

def create_hf_model(model_name:str):
    if torch_device == "cuda":
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")#, torch_dtype=torch.float16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")#, torch_dtype=torch.float16)
    else:
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)#, torch_dtype=torch.float16)
        else: 
            model = AutoModelForCausalLM.from_pretrained(model_name)#, trust_remote_code=True)#, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = context_length)

    return model, tokenizer

def load_model(model_type:str, gpu_layers:int, gpu_config:dict=gpu_config, cpu_config:dict=cpu_config, torch_device:str=torch_device):
    print("Loading model")

    if model_type == "Phi 3.5 Mini (larger, slow)":
        if torch_device == "cuda":
            gpu_config.update_gpu(gpu_layers)
            print("Loading with", gpu_config.n_gpu_layers, "model layers sent to GPU.")
        else:
            gpu_config.update_gpu(gpu_layers)
            cpu_config.update_gpu(gpu_layers)

            print("Loading with", cpu_config.n_gpu_layers, "model layers sent to GPU.")

        print(vars(gpu_config))
        print(vars(cpu_config))

        try:
            model = Llama(
            model_path=hf_hub_download(
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3.5-mini-instruct-GGUF"),# "QuantFactory/Phi-3-mini-128k-instruct-GGUF"), # "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3.5-mini-instruct.Q4_K_M.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")  #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf")#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(gpu_config) # change n_gpu_layers if you have more or less VRAM 
        )
        
        except Exception as e:
            print("GPU load failed", e)
            model = Llama(
            model_path=hf_hub_download(
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3.5-mini-instruct-GGUF"), #"QuantFactory/Phi-3-mini-128k-instruct-GGUF"), #, "microsoft/Phi-3-mini-4k-instruct-gguf"),#"QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3.5-mini-instruct.Q4_K_M.gguf"), # "Phi-3-mini-128k-instruct.Q4_K_M.gguf") # , #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf"),#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(cpu_config)
        )

        tokenizer = []

    if model_type == "Qwen 2 0.5B (small, fast)":
        # Huggingface chat model
        hf_checkpoint = 'Qwen/Qwen2-0.5B-Instruct'# 'declare-lab/flan-alpaca-large'#'declare-lab/flan-alpaca-base' # # # 'Qwen/Qwen1.5-0.5B-Chat' #
        
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
    model_type = "Qwen 2 0.5B (small, fast)"
    load_model(model_type, 0, gpu_config, cpu_config, torch_device) # chatf.model_object, chatf.tokenizer, chatf.model_type = 

    print("chatf.model_object:", chatf.model_object)

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
    embeddings_state = gr.State(chatf.embeddings)#globals()["embeddings"])
    vectorstore_state = gr.State(chatf.vectorstore)#globals()["vectorstore"]) 

    relevant_query_state = gr.Checkbox(value=True, visible=False) 

    model_state = gr.State() # chatf.model_object (gives error)
    tokenizer_state = gr.State() # chatf.tokenizer (gives error)

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    session_hash_state = gr.State()
    s3_output_folder_state = gr.State()

    session_hash_textbox = gr.Textbox(value="", visible=False)
    s3_logs_output_textbox = gr.Textbox(label="S3 logs", visible=False)

    access_logs_state = gr.State(access_logs_data_folder + 'dataset1.csv')
    access_s3_logs_loc_state = gr.State(access_logs_data_folder)
    usage_logs_state = gr.State(usage_data_folder + 'dataset1.csv')
    usage_s3_logs_loc_state = gr.State(usage_data_folder)
    feedback_logs_state = gr.State(feedback_data_folder + 'dataset1.csv')
    feedback_s3_logs_loc_state = gr.State(feedback_data_folder)

    gr.Markdown("<h1><center>Lightweight PDF / web page QA bot</center></h1>")        
    
    gr.Markdown("Chat with PDF, web page or (new) csv/Excel documents. The default is a small model (Qwen 2 0.5B), that can only answer specific questions that are answered in the text. It cannot give overall impressions of, or summarise the document. The alternative (Phi 3.5 Mini (larger, slow)), can reason a little better, but is much slower (See Advanced tab).\n\nBy default the Lambeth Borough Plan '[Lambeth 2030 : Our Future, Our Lambeth](https://www.lambeth.gov.uk/better-fairer-lambeth/projects/lambeth-2030-our-future-our-lambeth)' is loaded. If you want to talk about another document or web page, please select from the second tab. If switching topic, please click the 'Clear chat' button.\n\nCaution: This is a public app. Please ensure that the document you upload is not sensitive is any way as other users may see it! Also, please note that LLM chatbots may give incomplete or incorrect information, so please use with care.")

    with gr.Accordion(label="Use Gemini or AWS Claude model", open=False, visible=False):
        api_model_choice = gr.Dropdown(value = "None", choices = ["gemini-2.0-flash-001", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0", "None"], label="LLM model to use", multiselect=False, interactive=True, visible=False)
        in_api_key = gr.Textbox(value = "", label="Enter Gemini API key (only if using Google API models)", lines=1, type="password",interactive=True, visible=False)

    with gr.Row():
        current_source = gr.Textbox(label="Current data source(s)", value="Lambeth_2030-Our_Future_Our_Lambeth.pdf", scale = 10)
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
            stop = gr.Button(value="Stop generating", variant="secondary", scale=1)

        examples_set = gr.Radio(label="Examples for the Lambeth Borough Plan",
            #value = "What were the five pillars of the previous borough plan?",
            choices=["What were the five pillars of the previous borough plan?",
                "What is the vision statement for Lambeth?",
                "What are the commitments for Lambeth?",
                "What are the 2030 outcomes for Lambeth?"])
        
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
            model_choice = gr.Radio(label="Choose a chat model", value="Qwen 2 0.5B (small, fast)", choices = ["Qwen 2 0.5B (small, fast)", "Phi 3.5 Mini (larger, slow)", "gemini-2.0-flash-001", "gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25", "anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-sonnet-20240229-v1:0"])
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=100, step = 1, visible=True)
            
        load_text = gr.Text(label="Load status")        

    gr.HTML(
        "<center>This app is based on the models Qwen 2 0.5B and Phi 3.5 Mini. It powered by Gradio, Transformers, and Llama.cpp.</a></center>"
    )

    examples_set.change(fn=chatf.update_message, inputs=[examples_set], outputs=[message])


    ###
    # CHAT PAGE
    ###

    # Click to send message
    response_click = submit.click(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state, out_passages, api_model_choice, in_api_key], outputs=[chat_history_state, sources, instruction_prompt_out, relevant_query_state], queue=False, api_name="retrieval").\
                success(chatf.turn_off_interactivity, inputs=None, outputs=[message, submit], queue=False).\
                success(chatf.produce_streaming_answer_chatbot, inputs=[chatbot, instruction_prompt_out, model_type_state, temp_slide, relevant_query_state, chat_history_state], outputs=chatbot)
    response_click.success(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                success(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                success(lambda: chatf.restore_interactivity(), None, [message, submit], queue=False)

    # Press enter to send message
    response_enter = message.submit(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state, out_passages, api_model_choice, in_api_key], outputs=[chat_history_state, sources, instruction_prompt_out, relevant_query_state], queue=False).\
                success(chatf.turn_off_interactivity, inputs=None, outputs=[message, submit], queue=False).\
                success(chatf.produce_streaming_answer_chatbot, [chatbot, instruction_prompt_out, model_type_state, temp_slide, relevant_query_state, chat_history_state], chatbot)
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
             success(embed_faiss_save_to_zip, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             success(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage
    load_web_click = load_web.click(ing.parse_html, inputs=[in_web, in_div], outputs=[ingest_text, ingest_metadata, current_source]).\
             success(ing.html_text_to_docs, inputs=[ingest_text, ingest_metadata], outputs=[ingest_docs]).\
             success(embed_faiss_save_to_zip, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             success(chatf.hide_block, outputs = [examples_set])
    
    # Load in a csv/excel file
    load_csv_click = load_csv.click(ing.parse_csv_or_excel, inputs=[in_csv, in_text_column], outputs=[ingest_text, current_source]).\
             success(ing.csv_excel_text_to_docs, inputs=[ingest_text, in_text_column], outputs=[ingest_docs]).\
             success(embed_faiss_save_to_zip, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
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
    app.load(get_connection_params, inputs=None, outputs=[session_hash_state, s3_output_folder_state, session_hash_textbox]).\
    success(load_model, inputs=[model_type_state, gpu_layer_choice, gpu_config_state, cpu_config_state, torch_device_state], outputs=[model_type_state, load_text, current_model])

    # Log usernames and times of access to file (to know who is using the app when running on AWS)
    access_callback = gr.CSVLogger()
    access_callback.setup([session_hash_textbox], access_logs_data_folder)

    session_hash_textbox.change(lambda *args: access_callback.flag(list(args)), [session_hash_textbox], None, preprocess=False).\
    success(fn = upload_file_to_s3, inputs=[access_logs_state, access_s3_logs_loc_state], outputs=[s3_logs_output_textbox])

if __name__ == "__main__":
    if os.environ['COGNITO_AUTH'] == "1":
        app.queue().launch(show_error=True, auth=authenticate_user, max_file_size='50mb')
    else:
        app.queue().launch(show_error=True, inbrowser=True, max_file_size='50mb')