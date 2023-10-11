# # Load in packages

# +
import os
import copy

# Need to overwrite version of gradio present in Huggingface spaces as it doesn't have like buttons/avatars (Oct 2023)
#os.system("pip uninstall -y gradio")
os.system("pip install gradio==3.42.0")

from typing import TypeVar
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
import gradio as gr

from transformers import AutoTokenizer
from dataclasses import asdict, dataclass

# Alternative model sources
from ctransformers import AutoModelForCausalLM

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

# Disable cuda devices if necessary
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

#from chatfuncs.chatfuncs import *
import chatfuncs.ingest as ing


##  Load preset embeddings, vectorstore, and model

embeddings_name = "thenlper/gte-base"

def load_embeddings(embeddings_name = "thenlper/gte-base"):


    if embeddings_name == "hkunlp/instructor-large":
        embeddings_func = HuggingFaceInstructEmbeddings(model_name=embeddings_name,
        embed_instruction="Represent the paragraph for retrieval: ",
        query_instruction="Represent the question for retrieving supporting documents: "
        )

    else: 
        embeddings_func = HuggingFaceEmbeddings(model_name=embeddings_name)

    global embeddings

    embeddings = embeddings_func

    return embeddings

def get_faiss_store(faiss_vstore_folder,embeddings):
    import zipfile
    with zipfile.ZipFile(faiss_vstore_folder + '/' + faiss_vstore_folder + '.zip', 'r') as zip_ref:
        zip_ref.extractall(faiss_vstore_folder)

    faiss_vstore = FAISS.load_local(folder_path=faiss_vstore_folder, embeddings=embeddings)
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

    if model_type == "Orca Mini":

        gpu_config.update_gpu(gpu_layers)
        cpu_config.update_gpu(gpu_layers)

        print("Loading with", cpu_config.gpu_layers, "model layers sent to GPU.")

        print(vars(gpu_config))
        print(vars(cpu_config))

        try:
            model = AutoModelForCausalLM.from_pretrained('juanjgit/orca_mini_3B-GGUF', model_type='llama', model_file='orca-mini-3b.q4_0.gguf', **vars(cpu_config)) # **asdict(CtransRunConfig_cpu())
        except:
            model = AutoModelForCausalLM.from_pretrained('juanjgit/orca_mini_3B-GGUF', model_type='llama', model_file='orca-mini-3b.q4_0.gguf', **vars(gpu_config)) #**asdict(CtransRunConfig_gpu())

        tokenizer = []

    if model_type == "Flan Alpaca":
        # Huggingface chat model
        hf_checkpoint = 'declare-lab/flan-alpaca-large'
        
        def create_hf_model(model_name):

            from transformers import AutoModelForSeq2SeqLM,  AutoModelForCausalLM
            
            if torch_device == "cuda":
                if "flan" in model_name:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
            else:
                if "flan" in model_name:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                else: 
                    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

            tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = chatf.context_length)

            return model, tokenizer, model_type

        model, tokenizer, model_type = create_hf_model(model_name = hf_checkpoint)

    chatf.model = model
    chatf.tokenizer = tokenizer
    chatf.model_type = model_type

    print("Finished loading model: ", model_type)
    return model_type

# Both models are loaded on app initialisation so that users don't have to wait for the models to be downloaded
model_type = "Orca Mini"

load_model(model_type, chatf.gpu_layers, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

model_type = "Flan Alpaca"
load_model(model_type, 0, chatf.gpu_config, chatf.cpu_config, chatf.torch_device)

def docs_to_faiss_save(docs_out:PandasDataFrame, embeddings=embeddings):

    print(f"> Total split documents: {len(docs_out)}")

    print(docs_out)

    vectorstore_func = FAISS.from_documents(documents=docs_out, embedding=embeddings)


    chatf.vectorstore = vectorstore_func

    out_message = "Document processing complete"

    return out_message, vectorstore_func

 # Gradio chat

block = gr.Blocks(theme = gr.themes.Base())#css=".gradio-container {background-color: black}")

with block:
    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()

    model_type_state = gr.State(model_type)
    embeddings_state = gr.State(globals()["embeddings"])
    vectorstore_state = gr.State(globals()["vectorstore"])  

    model_state = gr.State() # chatf.model (gives error)
    tokenizer_state = gr.State() # chatf.tokenizer (gives error)

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    gr.Markdown("<h1><center>Lightweight PDF / web page QA bot</center></h1>")        
    
    gr.Markdown("Chat with PDF or web page documents. The default is a small model (Flan Alpaca), that can only answer specific questions that are answered in the text. It cannot give overall impressions of, or summarise the document. The alternative (Orca Mini), can reason a little better, but is much slower (See Advanced tab).\n\nBy default the Lambeth Borough Plan '[Lambeth 2030 : Our Future, Our Lambeth](https://www.lambeth.gov.uk/better-fairer-lambeth/projects/lambeth-2030-our-future-our-lambeth)' is loaded. If you want to talk about another document or web page, please select from the second tab. If switching topic, please click the 'Clear chat' button.\n\nCaution: This is a public app. Likes and dislike responses will be saved to disk to improve the model. Please ensure that the document you upload is not sensitive is any way as other users may see it! Also, please note that LLM chatbots may give incomplete or incorrect information, so please use with care.")

    current_source = gr.Textbox(label="Current data source that is loaded into the app", value="Lambeth_2030-Our_Future_Our_Lambeth.pdf")

    with gr.Tab("Chatbot"):

        with gr.Row():
            chat_height = 500
            chatbot = gr.Chatbot(height=chat_height, avatar_images=('user.jfif', 'bot.jpg'),bubble_full_width = False)
            sources = gr.HTML(value = "Source paragraphs where I looked for answers will appear here", height=chat_height)

        with gr.Row():
            message = gr.Textbox(
                label="What's your question?",
                lines=1,
            )     
        with gr.Row():
            submit = gr.Button(value="Send message", variant="secondary", scale = 1)
            clear = gr.Button(value="Clear chat", variant="secondary", scale=0)  

        examples_set = gr.Radio(label="Examples for the Lambeth Borough Plan",
            #value = "What were the five pillars of the previous borough plan?",
            choices=["What were the five pillars of the previous borough plan?",
                "What is the vision statement for Lambeth?",
                "What are the commitments for Lambeth?",
                "What are the 2030 outcomes for Lambeth?"])

        
        current_topic = gr.Textbox(label="Feature currently disabled - Keywords related to current conversation topic.", placeholder="Keywords related to the conversation topic will appear here")
            


    with gr.Tab("Load in a different PDF file or web page to chat"):
        with gr.Accordion("PDF file", open = False):
            in_pdf = gr.File(label="Upload pdf", file_count="multiple", file_types=['.pdf'])
            load_pdf = gr.Button(value="Load in file", variant="secondary", scale=0)
        
        with gr.Accordion("Web page", open = False):
            with gr.Row():
                in_web = gr.Textbox(label="Enter webpage url")
                in_div = gr.Textbox(label="(Advanced) Webpage div for text extraction", value="p", placeholder="p")
            load_web = gr.Button(value="Load in webpage", variant="secondary", scale=0) 
        
        ingest_embed_out = gr.Textbox(label="File/webpage preparation progress")

    with gr.Tab("Advanced features"):
        model_choice = gr.Radio(label="Choose a chat model", value="Flan Alpaca", choices = ["Flan Alpaca", "Orca Mini"])
        gpu_layer_choice = gr.Slider(label="Choose number of model layers to send to GPU (please don't change if you don't know what you're doing).", value=0, minimum=0, maximum=6, step = 1, scale = 0)

    gr.HTML(
        "<center>This app is based on the models Flan Alpaca and Orca Mini. It powered by Gradio, Transformers, Ctransformers, and Langchain.</a></center>"
    )

    examples_set.change(fn=chatf.update_message, inputs=[examples_set], outputs=[message])

    model_choice.change(fn=load_model, inputs=[model_choice, gpu_layer_choice], outputs = [model_type_state])

    # Load in a pdf
    load_pdf_click = load_pdf.click(ing.parse_file, inputs=[in_pdf], outputs=[ingest_text, current_source]).\
             then(ing.text_to_docs, inputs=[ingest_text], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state]).\
             then(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage
    load_web_click = load_web.click(ing.parse_html, inputs=[in_web, in_div], outputs=[ingest_text, ingest_metadata, current_source]).\
             then(ing.html_text_to_docs, inputs=[ingest_text, ingest_metadata], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=[ingest_embed_out, vectorstore_state]).\
             then(chatf.hide_block, outputs = [examples_set])

    # Load in a webpage

    # Click/enter to send message action
    response_click = submit.click(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False, api_name="retrieval").\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot, inputs=[chatbot, instruction_prompt_out, model_type_state], outputs=chatbot)
    response_click.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: chatf.restore_interactivity(), None, [message], queue=False)

    response_enter = message.submit(chatf.create_full_prompt, inputs=[message, chat_history_state, current_topic, vectorstore_state, embeddings_state, model_type_state], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False).\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot, [chatbot, instruction_prompt_out, model_type_state], chatbot)    
    response_enter.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: chatf.restore_interactivity(), None, [message], queue=False)
    
    # Clear box
    clear.click(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic])
    clear.click(lambda: None, None, chatbot, queue=False)

    chatbot.like(chatf.vote, [chat_history_state, instruction_prompt_out, model_type_state], None)

block.queue(concurrency_count=1).launch(debug=True)
# -

