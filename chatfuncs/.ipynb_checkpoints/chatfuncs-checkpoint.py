# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import datetime
from typing import Dict, List, Tuple
from itertools import compress
import pandas as pd

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate
from langchain.retrievers import TFIDFRetriever, SVMRetriever
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
#from transformers import pipeline
from optimum.pipelines import pipeline
from transformers import AutoTokenizer, TextStreamer, AutoModelForSeq2SeqLM, TextIteratorStreamer
from threading import Thread

import gradio as gr


# -

# # Pre-load stopwords, vectorstore, models

# +
def get_faiss_store(faiss_vstore_folder,embeddings):
    import zipfile
    with zipfile.ZipFile(faiss_vstore_folder + '/faiss_lambeth_census_embedding.zip', 'r') as zip_ref:
        zip_ref.extractall(faiss_vstore_folder)

    faiss_vstore = FAISS.load_local(folder_path=faiss_vstore_folder, embeddings=embeddings)
    os.remove(faiss_vstore_folder + "/index.faiss")
    os.remove(faiss_vstore_folder + "/index.pkl")
    
    return faiss_vstore

#def set_hf_api_key(api_key, chain_agent):
    #if api_key:
       #os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        #vectorstore = get_faiss_store(faiss_vstore_folder="faiss_lambeth_census_embedding.zip",embeddings=embeddings)
        #qa_chain = create_prompt_templates(vectorstore)
        #print(qa_chain)
        #os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""
        #return qa_chain


# -

def create_hf_model(model_name = "declare-lab/flan-alpaca-large"):

    model_id = model_name
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on device:", torch_device)
    print("CPU threads:", torch.get_num_threads())
    


    if torch_device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")
    else:
        #torch.set_num_threads(8)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer, torch_device

# +
# Add some stopwords to nltk default

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
#print(stopwords.words('english'))
newStopWords = ['what','how', 'when', 'which', 'who', 'change', 'changed', 'do', 'did', 'increase', 'decrease', 'increased',
                'decreased', 'proportion', 'percentage', 'report', 'reporting','say', 'said']
stopwords.extend(newStopWords)
# -

# Embeddings
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
#embeddings = HuggingFaceEmbeddings(model_name=model_name)
embed_model_name = "hkunlp/instructor-large"
embeddings = HuggingFaceInstructEmbeddings(model_name=embed_model_name)
vectorstore = get_faiss_store(faiss_vstore_folder="faiss_lambeth_census_embedding",embeddings=embeddings)

# +
# Models

#checkpoint = 'declare-lab/flan-alpaca-base' # Flan Alpaca Base incorrectly interprets text based on input (e.g. if you use words like increase or decrease in the question it will respond falsely often). Flan Alpaca Large is much more consistent
checkpoint = 'declare-lab/flan-alpaca-large'

model, tokenizer, torch_device = create_hf_model(model_name = checkpoint)


# Look at this for streaming text with huggingface and langchain (last example): https://github.com/hwchase17/langchain/issues/2918

streamer = TextStreamer(tokenizer, skip_prompt=True)

pipe = pipeline('text2text-generation', 
                 model = checkpoint,
#                tokenizer = tokenizer,
                 max_length=512, 
                 #do_sample=True,
                 temperature=0.000001,
                 #top_p=0.95,
                 #repetition_penalty=1.15,
                 accelerator="bettertransformer",
                 streamer=streamer
                )

checkpoint_keywords = 'ml6team/keyphrase-generation-t5-small-inspec'

keyword_model = pipeline('text2text-generation', 
                 model = checkpoint_keywords,
                 accelerator="bettertransformer"
                )


# -

# # Chat history

def clear_chat(chat_history_state, sources, chat_message):
    chat_history_state = []
    sources = ''
    chat_message = ''
    return chat_history_state, sources, chat_message


def _get_chat_history(chat_history: List[Tuple[str, str]]): # Limit to last 3 interactions only
    max_chat_length = 3

    if len(chat_history) > max_chat_length:
        chat_history = chat_history[-max_chat_length:]
        
    print(chat_history)

    first_q = ""
    for human_s, ai_s in chat_history:
        first_q = human_s
        break

    conversation = ""
    for human_s, ai_s in chat_history:
        human = f"Human: " + human_s
        ai = f"Assistant: " + ai_s
        conversation += "\n" + "\n".join([human, ai])

    return conversation, first_q


def adapt_q_from_chat_history(keyword_model, new_question_keywords, question, chat_history):
        t5_small_keyphrase = HuggingFacePipeline(pipeline=keyword_model)
        memory_llm = t5_small_keyphrase#flan_alpaca#flan_t5_xxl
        new_q_memory_llm = t5_small_keyphrase#flan_alpaca#flan_t5_xxl
    

        memory_prompt = PromptTemplate(
            template = "{chat_history_first_q}",
            input_variables=["chat_history_first_q"]
        )
            #template = "Extract the names of people, things, or places from the following text: {chat_history}",#\n    Original question: {question}\n  New list:",
            #template = "Extract keywords, and the names of people or places from the following text: {chat_history}",#\n    Original question: {question}\n  New list:",
            #\n    Original question: {question}\n  New list:",
        

            #example_prompt=_eg_prompt,
            #input_variables=["question", "chat_history"]
            #input_variables=["chat_history"]
        
        memory_extractor = LLMChain(llm=memory_llm, prompt=memory_prompt)
       
        #new_question_keywords = #remove_stopwords(question)

        print("new_question_keywords:")
        print(new_question_keywords)

        chat_history_str, chat_history_first_q = _get_chat_history(chat_history)
        if chat_history_str:
            
            extracted_memory = memory_extractor.run(
                chat_history_first_q=chat_history_first_q # question=question, chat_history=chat_history_str, 
            )
            
            new_question_kworded = extracted_memory + " " + new_question_keywords
            new_question = extracted_memory + " " + question
            
        else:
            new_question = question
            new_question_kworded = new_question_keywords
            
        return new_question, new_question_kworded


# # Prompt creation

def remove_q_stopwords(question):
    # Prepare question by removing keywords
    text = question.lower()
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    new_question_keywords = ' '.join(tokens_without_sw)
    return new_question_keywords, question


def create_final_prompt(inputs: Dict[str, str], vectorstore, instruction_prompt, content_prompt):
        
        question =  inputs["question"]
        chat_history = inputs["chat_history"]
        
        new_question_keywords, question = remove_q_stopwords(question)

        new_question, new_question_kworded = adapt_q_from_chat_history(keyword_model, new_question_keywords, question, chat_history)
        

        print("The question passed to the vector search is:")
        print(new_question_kworded)
        
        docs_keep_as_doc, docs_content, docs_url = find_relevant_passages(new_question_kworded, embeddings, k_val = 3, out_passages = 2, vec_score_cut_off = 1.3, vec_weight = 1, tfidf_weight = 0.5, svm_weight = 1)

        if docs_keep_as_doc == []:
            {"answer": "I'm sorry, I couldn't find a relevant answer to this question.", "sources":"I'm sorry, I couldn't find a relevant source for this question."}
        
        #new_inputs = inputs.copy()
        #new_inputs["question"] = new_question
        #new_inputs["chat_history"] = chat_history_str
        
        string_docs_content = '\n\n\n'.join(docs_content)
        
        #print("The draft instruction prompt is:")
        #print(instruction_prompt)
        
        instruction_prompt_out = instruction_prompt.format(question=new_question, summaries=string_docs_content)
        #print("The final instruction prompt:")
        #print(instruction_prompt_out)
        
                
        return instruction_prompt_out, string_docs_content


# +
def create_prompt_templates():    
  
    #EXAMPLE_PROMPT = PromptTemplate(
    #    template="\nCONTENT:\n\n{page_content}\n\nSOURCE: {source}\n\n",
    #    input_variables=["page_content", "source"],
    #)

    CONTENT_PROMPT = PromptTemplate(
        template="{page_content}\n\n",#\n\nSOURCE: {source}\n\n",
        input_variables=["page_content"]
    )


# The main prompt:

    #main_prompt_template = """
    #Answer the question using the CONTENT below:  

    #CONTENT: {summaries}
    
    #QUESTION: {question}

    #ANSWER: """

    instruction_prompt_template = """
    {summaries}
    
    QUESTION: {question}
    
    Quote relevant text above."""

   
    INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_template, input_variables=['question', 'summaries'])
    
    return INSTRUCTION_PROMPT, CONTENT_PROMPT


# -

def get_history_sources_final_input_prompt(user_input, history):
    
    #if chain_agent is None:
    #    history.append((user_input, "Please click the button to submit the Huggingface API key before using the chatbot (top right)"))
    #    return history, history, "", ""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("User input: " + user_input)
    
    history = history or []
    

    
    # Create instruction prompt
    instruction_prompt, content_prompt = create_prompt_templates()
    instruction_prompt_out, string_docs_content =\
                create_final_prompt({"question": user_input, "chat_history": history}, vectorstore,
                                    instruction_prompt, content_prompt)
    
    sources_txt =  string_docs_content
    
    #print('sources_txt:')
    #print(sources_txt)
    
    history.append(user_input)
    
    print("Output history is:")
    print(history)

    print("The output prompt is:")
    print(instruction_prompt_out)
    
    return history, sources_txt, instruction_prompt_out


# # Chat functions

def produce_streaming_answer_chatbot(history, full_prompt): 
    
    print("The question is: ")
    print(full_prompt)
    
    # Get the model and tokenizer, and tokenize the user text.
    model_inputs = tokenizer(text=full_prompt, return_tensors="pt").to(torch_device)

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        #top_p=top_p,
        temperature=float(0.00001)#,
        #top_k=top_k
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    
    history[-1][1] = ""
    for new_text in streamer:
        history[-1][1] += new_text
        yield history


def user(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]


def add_inputs_answer_to_history(user_message, history):
    #history.append((user_message, [-1]))
    
    print("History after appending is:")
    print(history)
    
    
    return history


# # Vector / hybrid search

def find_relevant_passages(new_question_kworded, embeddings, k_val, out_passages, vec_score_cut_off, vec_weight, tfidf_weight, svm_weight, vectorstore=vectorstore):

            docs = vectorstore.similarity_search_with_score(new_question_kworded, k=k_val)
            #docs = self.vstore.similarity_search_with_score(new_question_kworded, k=k_val)

            # Keep only documents with a certain score
            #docs_orig = [x[0] for x in docs]
            docs_scores = [x[1] for x in docs]

            # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
            score_more_limit = pd.Series(docs_scores) < vec_score_cut_off
            docs_keep = list(compress(docs, score_more_limit))

            if docs_keep == []:
                docs_keep_as_doc = []
                docs_content = []
                docs_url = []
                return docs_keep_as_doc, docs_content, docs_url
            
            

            docs_keep_as_doc = [x[0] for x in docs_keep]
            docs_keep_length = len(docs_keep_as_doc)

            #print('docs_keep:')
            #print(docs_keep)

            vec_rank = [*range(1, docs_keep_length+1)]
            vec_score = [(docs_keep_length/x)*vec_weight for x in vec_rank]

            #print("vec_rank")
            #print(vec_rank)

            #print("vec_score")
            #print(vec_score)

        

            # 2nd level check on retrieved docs with TFIDF
            content_keep=[]
            for item in docs_keep:
                content_keep.append(item[0].page_content)

            tfidf_retriever = TFIDFRetriever.from_texts(content_keep, k = k_val)
            tfidf_result = tfidf_retriever.get_relevant_documents(new_question_kworded)

            #print("TDIDF retriever result:")
            #print(tfidf_result)

            tfidf_rank=[]
            tfidf_score = []

            for vec_item in docs_keep:
                x = 0
                for tfidf_item in tfidf_result:
                    x = x + 1
                    if tfidf_item.page_content == vec_item[0].page_content:
                        tfidf_rank.append(x)
                        tfidf_score.append((docs_keep_length/x)*tfidf_weight)

            #print("tfidf_rank:")
            #print(tfidf_rank)
            #print("tfidf_score:")
            #print(tfidf_score)


            # 3rd level check on retrieved docs with SVM retriever
            svm_retriever = SVMRetriever.from_texts(content_keep, embeddings, k = k_val)
            svm_result = svm_retriever.get_relevant_documents(new_question_kworded)

            #print("SVM retriever result:")
            #print(svm_result)
         
            svm_rank=[]
            svm_score = []

            for vec_item in docs_keep:
                x = 0
                for svm_item in svm_result:
                    x = x + 1
                    if svm_item.page_content == vec_item[0].page_content:
                        svm_rank.append(x)
                        svm_score.append((docs_keep_length/x)*svm_weight)

            #print("svm_score:")
            #print(svm_score)

        
            ## Calculate final score based on three ranking methods
            final_score = [a  + b + c for a, b, c in zip(vec_score, tfidf_score, svm_score)]
            final_rank = [sorted(final_score, reverse=True).index(x)+1 for x in final_score]

            #print("Final score:")
            #print(final_score)
            #print("final rank:")
            #print(final_rank)

            best_rank_index_pos = []

            for x in range(1,out_passages+1):
                try:
                    best_rank_index_pos.append(final_rank.index(x))
                except IndexError: # catch the error
                    pass

            # Adjust best_rank_index_pos to 
        
            #print("Best rank positions in original vector search list:")
            #print(best_rank_index_pos)

            best_rank_pos_series = pd.Series(best_rank_index_pos)
            #docs_keep_out = list(compress(docs_keep, best_rank_pos_series))

            #print("docs_keep:")
            #print(docs_keep)

            docs_keep_out = [docs_keep[i] for i in best_rank_index_pos]
        

            #docs_keep = [(docs_keep[best_rank_pos])]
            # Keep only 'best' options
            docs_keep_as_doc = [x[0] for x in docs_keep_out]# [docs_keep_as_doc_filt[0]]#[x[0] for x in docs_keep_as_doc_filt] #docs_keep_as_doc_filt[0]#
        
            #print("docs_keep_out:")
            #print(docs_keep_out)

            # Extract content and metadata from 'winning' passages.

            content=[]
            meta_url=[]
            score=[]

            for item in docs_keep_out:
                content.append(item[0].page_content)
                meta_url.append(item[0].metadata['source'])
                score.append(item[1])       

            # Create df from 'winning' passages

            doc_df = pd.DataFrame(list(zip(content, meta_url, score)),
               columns =['page_content', 'meta_url', 'score'])#.iloc[[0, 1]]
        
            #print("docs_keep_as_doc: ")
            #print(docs_keep_as_doc)

            #print("doc_df")
            #print(doc_df)

            docs_content = doc_df['page_content'].astype(str)
            docs_url = "https://" + doc_df['meta_url']
        
            #print("Docs meta url is: ")
            #print(docs_meta_url)

            #print("Docs content is: ")
            #print(docs_content)

            #docs_url = [d['source'] for d in docs_meta]
            #print(docs_url)
            
            

            return docs_keep_as_doc, docs_content, docs_url
