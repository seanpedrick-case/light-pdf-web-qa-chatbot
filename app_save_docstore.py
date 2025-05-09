# Load in packages

import os

from typing import Type
from langchain_community.embeddings import HuggingFaceEmbeddings#, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
import pandas as pd

from transformers import AutoTokenizer
import torch

from llama_cpp import Llama
from huggingface_hub import hf_hub_download  

PandasDataFrame = Type[pd.DataFrame]

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

#from chatfuncs.chatfuncs import *
import chatfuncs.ingest as ing

##  Load preset embeddings, vectorstore, and model

embeddings_name = "BAAI/bge-base-en-v1.5"

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

chatf.embeddings = load_embeddings(embeddings_name)
chatf.vectorstore = get_faiss_store(faiss_vstore_folder="faiss_embedding",embeddings=globals()["embeddings"])


def load_model(model_type, gpu_layers, gpu_config=None, cpu_config=None, torch_device=None):
    print("Loading model")

    # Default values inside the function
    if gpu_config is None:
        gpu_config = chatf.gpu_config
    if cpu_config is None:
        cpu_config = chatf.cpu_config
    if torch_device is None:
        torch_device = chatf.torch_device

    if model_type == "Phi 3 Mini (larger, slow)":
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
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3-mini-128k-instruct-GGUF"),# "QuantFactory/Phi-3-mini-128k-instruct-GGUF"), # "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3-mini-128k-instruct.Q4_K_M.gguf") #"Phi-3-mini-128k-instruct.Q4_K_M.gguf")  #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf")#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(gpu_config) # change n_gpu_layers if you have more or less VRAM 
        )
        
        except Exception as e:
            print("GPU load failed")
            print(e)
            model = Llama(
            model_path=hf_hub_download(
            repo_id=os.environ.get("REPO_ID", "QuantFactory/Phi-3-mini-128k-instruct-GGUF"), #"QuantFactory/Phi-3-mini-128k-instruct-GGUF"), #, "microsoft/Phi-3-mini-4k-instruct-gguf"),#"QuantFactory/Meta-Llama-3-8B-Instruct-GGUF-v2"), #"microsoft/Phi-3-mini-4k-instruct-gguf"),#"TheBloke/Mistral-7B-OpenOrca-GGUF"),
            filename=os.environ.get("MODEL_FILE", "Phi-3-mini-128k-instruct.Q4_K_M.gguf"), # "Phi-3-mini-128k-instruct.Q4_K_M.gguf") # , #"Meta-Llama-3-8B-Instruct-v2.Q6_K.gguf") #"Phi-3-mini-4k-instruct-q4.gguf"),#"mistral-7b-openorca.Q4_K_M.gguf"),
        ),
        **vars(cpu_config)
        )

        tokenizer = []

    if model_type == "Flan Alpaca (small, fast)":
        # Huggingface chat model
        hf_checkpoint = 'declare-lab/flan-alpaca-large'#'declare-lab/flan-alpaca-base' # # #
        
        def create_hf_model(model_name):

            from transformers import AutoModelForSeq2SeqLM,  AutoModelForCausalLM
            
            if torch_device == "cuda":
                if "flan" in model_name:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
            else:
                if "flan" in model_name:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16)
                else: 
                    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)

            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = chatf.context_length)

            return model, tokenizer, model_type

        model, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    chatf.model = model
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    load_confirmation = "Finished loading model: " + model_type

    print(load_confirmation)
    return model_type, load_confirmation, model_type

# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
model_type = "Phi 3 Mini (larger, slow)"
load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Flan Alpaca (small, fast)"
load_model(model_type, 0, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    print(docs_out)

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)


    chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, vectorstore_func, out_file

 # Gradio chat

block = gr.Blocks(theme = gr.themes.Base())#css=".gradio-container {background-color: black}")

with block:
    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()

    model_type_state = gr.State(model_type)
    embeddings_state = gr.State(chatf.embeddings)#globals()["embeddings"])
    vectorstore_state = gr.State(chatf.vectorstore)#globals()["vectorstore"])  

    model_state = gr.State() # chatf.model (gives error)
    tokenizer_state = gr.State() # chatf.tokenizer (gives error)

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    gr.Markdown("<h1><center>Lightweight PDF / web page QA bot</center></h1>")        
    
    gr.Markdown("Chat with PDF, web page or (new) csv/Excel documents. The default is a small model (Flan Alpaca), that can only answer specific questions that are answered in the text. It cannot give overall impressions of, or summarise the document. The alternative (Phi 3 Mini (larger, slow)), can reason a little better, but is much slower (See Advanced tab).\n\nBy default the Lambeth Borough Plan '[Lambeth 2030 : Our Future, Our Lambeth](https://www.lambeth.gov.uk/better-fairer-lambeth/projects/lambeth-2030-our-future-our-lambeth)' is loaded. If you want to talk about another document or web page, please select from the second tab. If switching topic, please click the 'Clear chat' button.\n\nCaution: This is a public app. Please ensure that the document you upload is not sensitive is any way as other users may see it! Also, please note that LLM chatbots may give incomplete or incorrect information, so please use with care.")

    with gr.Row():
        current_source = gr.Textbox(label="Current data source(s)", value="Lambeth_2030-Our_Future_Our_Lambeth.pdf", scale = 10)
        current_model = gr.Textbox(label="Current model", value=model_type, scale = 3)

    with gr.Tab("Chatbot"):

        with gr.Row():
            #chat_height = 500
            chatbot = gr.Chatbot(avatar_images=('user.jfif', 'bot.jpg'),bubble_full_width = False, scale = 1) # , height=chat_height
            with gr.Accordion("Open this tab to see the source paragraphs used to generate the answer", open = False):
                sources = gr.HTML(value = "Source paragraphs with the most relevant text will appear here") # , height=chat_height

        with gr.Row():
            message = gr.Textbox(
                label="Enter your question here",
                lines=1,
            )     
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary", scale = 1)
            clear = gr.Button(value="Clear chat", variant="secondary", scale=0) 
            stop = gr.Button(value="Stop generating", variant="secondary", scale=0)

        examples_set = gr.Radio(label="Examples for the Lambeth Borough Plan",
            #value = "What were the five pillars of the previous borough plan?",
            choices=["What were the five pillars of the previous borough plan?",
                "What is the vision statement for Lambeth?",
                "What are the commitments for Lambeth?",
                "What are the 2030 outcomes for Lambeth?"])

        
        current_topic = gr.Textbox(label="Feature currently disabled - Keywords related to current conversation topic.", placeholder="Keywords related to the conversation topic will appear here")      


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
        	out_file_box = gr.File(count='single', filetype=['.zip'])

    with gr.Tab("Advanced features"):
        out_passages = gr.Slider(minimum=1, value = 2, maximum=10, step=1, label="Choose number of passages to retrieve from the document. Numbers greater than 2 may lead to increased hallucinations or input text being truncated.")
        temp_slide = gr.Slider(minimum=0.1, value = 0.5, maximum=1, step=0.1, label="Choose temperature setting for response generation.")
        with gr.Row():
            model_choice = gr.Radio(label="Choose a chat model", value="Flan Alpaca (small, fast)", choices = ["Flan Alpaca (small, fast)", "Phi 3 Mini (larger, slow)"])
            change_model_button = gr.Button(value="Load model", scale=0)
        with gr.Accordion("Choose number of model layers to send to GPU (WARNING: please don't modify unless you are sure you have a GPU).", open = False):
            gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU.", value=0, minimum=0, maximum=100, step = 1, visible=True)
            
        load_text = gr.Text(label="Load status")
        

    gr.HTML(
        "<center>This app is based on the models Flan Alpaca and Phi 3 Mini. It powered by Gradio, Transformers, and Llama.cpp.</a></center>"
    )

    examples_set.change(fn=chatf.update_message, inputs=[examples_set], outputs=[message])

    change_model_button.click(fn=chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
    then(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state, load_text, current_model]).\
    then(lambda: chatf.restore_interactivity(), None, [message], queue=False).\
    then(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic]).\
    then(lambda: None, None, chatbot, queue=False)

    # Load in a pdf
    load_pdf_click = load_pdf.click(ing.parse_file, inputs=[in_pdf], outputs=[ingest_text, current_source]).\
             then(ing.text_to_docs, inputs=[ingest_text], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             then(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage
    load_web_click = load_web.click(ing.parse_html, inputs=[in_web, in_div], outputs=[ingest_text, ingest_metadata, current_source]).\
             then(ing.html_text_to_docs, inputs=[ingest_text, ingest_metadata], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             then(chatf.hide_block, outputs = [examples_set])
    
    # Load in a csv/excel file
    load_csv_click = load_csv.click(ing.parse_csv_or_excel, inputs=[in_csv, in_text_column], outputs=[ingest_text, current_source]).\
             then(ing.csv_excel_text_to_docs, inputs=[ingest_text, in_text_column], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state, file_out_box]).\
             then(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage

    # Click/enter to send message action
    response_click = submit.click(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state, out_passages], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False, api_name="retrieval").\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot, inputs=[chatbot, instruction_prompt_out, model_type_state, temp_slide], outputs=chatbot)
    response_click.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: chatf.restore_interactivity(), None, [message], queue=False)

    response_enter = message.submit(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state, out_passages], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False).\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot, [chatbot, instruction_prompt_out, model_type_state, temp_slide], chatbot)    
    response_enter.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: chatf.restore_interactivity(), None, [message], queue=False)
    
    # Stop box
    stop.click(fn=None, inputs=None, outputs=None, cancels=[response_click, response_enter])

    # Clear box
    clear.click(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic])
    clear.click(lambda: None, None, chatbot, queue=False)

    # Thumbs up or thumbs down voting function
    chatbot.like(chatf.vote, [chat_history_state, instruction_prompt_out, model_type_state], None)

block.queue().launch(debug=True)