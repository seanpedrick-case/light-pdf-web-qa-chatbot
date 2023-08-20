# # Load in packages

# +
import os
from typing import TypeVar
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS


#PandasDataFrame: type[pd.core.frame.DataFrame]
PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

# Disable cuda devices if necessary

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

#from chatfuncs.chatfuncs import *
import chatfuncs.ingest as ing

##  Load preset embeddings and vectorstore

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

    chatf.vectorstore = vectorstore

    out_message = "Document processing complete"

    #print(out_message)
    #print(f"> Saved to: {save_to}")

    return out_message

 # Gradio chat

import gradio as gr

block = gr.Blocks(css=".gradio-container {background-color: black}")

with block:
    #with gr.Row():
    gr.Markdown("<h1><center>Lightweight PDF / web page QA bot</center></h1>")        
    
    gr.Markdown("By default the Lambeth Borough Plan '[Lambeth 2030 : Our Future, Our Lambeth](https://www.lambeth.gov.uk/better-fairer-lambeth/projects/lambeth-2030-our-future-our-lambeth)' is loaded. If you want to talk about another document or web page, please select below. The chatbot will not answer questions where answered can't be found on the website.\n\nIf switching topic, please click the 'New topic' button as the bot will assume follow up questions are linked to the first. Sources are shown underneath the chat area.")

    with gr.Tab("Chatbot"):

        with gr.Row():
            chatbot = gr.Chatbot(height=300)
            sources = gr.HTML(value = "Source paragraphs where I looked for answers will appear here", height=300)

        with gr.Row():
            message = gr.Textbox(
                label="What's your question?",
                lines=1,
            )     


        submit = gr.Button(value="Send message", variant="secondary", scale = 1)

        examples_set = gr.Examples(label="Examples for the Lambeth Borough Plan",
            examples=[
                "What were the five pillars of the previous borough plan?",
                "What is the vision statement for Lambeth?",
                "What are the commitments for Lambeth?",
                "What are the 2030 outcomes for Lambeth?"],
            inputs=message,
        )

        with gr.Row():
            current_topic = gr.Textbox(label="Current conversation topic. If you want to talk about something else, press 'New topic'", placeholder="Keywords related to the conversation topic will appear here")
            clear = gr.Button(value="New topic", variant="secondary", scale=0)  


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

    gr.HTML(
        "<center>Powered by Flan Alpaca and Langchain</a></center>"
    )

    ingest_text = gr.State()
    ingest_metadata = gr.State()
    ingest_docs = gr.State()

    #embeddings_state = gr.State()
    vectorstore_state = gr.State()

    chat_history_state = gr.State()
    instruction_prompt_out = gr.State()

    #def hide_examples():
    #    return gr.Examples.update(visible=False)

    # Load in a pdf
    load_pdf_click = load_pdf.click(ing.parse_file, inputs=[in_pdf], outputs=[ingest_text]).\
             then(ing.text_to_docs, inputs=[ingest_text], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=ingest_embed_out)
             #then(hide_examples)

    # Load in a webpage
    load_web_click = load_web.click(ing.parse_html, inputs=[in_web, in_div], outputs=[ingest_text, ingest_metadata]).\
             then(ing.html_text_to_docs, inputs=[ingest_text, ingest_metadata], outputs=[ingest_docs]).\
             then(docs_to_faiss_save, inputs=[ingest_docs], outputs=ingest_embed_out)
             #then(hide_examples)

    # Load in a webpage

    # Click/enter to send message action
    response_click = submit.click(chatf.get_history_sources_final_input_prompt, inputs=[message, chat_history_state, current_topic], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False).\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot_hf, inputs=[chatbot, instruction_prompt_out], outputs=chatbot)
    response_click.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: gr.update(interactive=True), None, [message], queue=False)

    response_enter = message.submit(chatf.get_history_sources_final_input_prompt, inputs=[message, chat_history_state, current_topic], outputs=[chat_history_state, sources, instruction_prompt_out], queue=False).\
                then(chatf.turn_off_interactivity, inputs=[message, chatbot], outputs=[message, chatbot], queue=False).\
                then(chatf.produce_streaming_answer_chatbot_hf, [chatbot, instruction_prompt_out], chatbot)    
    response_enter.then(chatf.highlight_found_text, [chatbot, sources], [sources]).\
                then(chatf.add_inputs_answer_to_history,[message, chatbot, current_topic], [chat_history_state, current_topic]).\
                then(lambda: gr.update(interactive=True), None, [message], queue=False)
    
    # Clear box
    clear.click(chatf.clear_chat, inputs=[chat_history_state, sources, message, current_topic], outputs=[chat_history_state, sources, message, current_topic])
    clear.click(lambda: None, None, chatbot, queue=False)

block.queue(concurrency_count=1).launch(debug=True)
# -
