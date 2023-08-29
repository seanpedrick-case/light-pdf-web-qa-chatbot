import re
import datetime
from typing import TypeVar, Dict, List, Tuple
from itertools import compress
import pandas as pd
import numpy as np

# Model packages
import torch
from threading import Thread
from transformers import AutoTokenizer, pipeline, TextIteratorStreamer

# Alternative model sources
from gpt4all import GPT4All
from ctransformers import AutoModelForCausalLM

from dataclasses import asdict, dataclass

# Langchain functions
from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.retrievers import SVMRetriever 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# For keyword extraction
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import keybert

#from transformers.pipelines import pipeline

# For Name Entity Recognition model
from span_marker import SpanMarkerModel

# For BM25 retrieval
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, OkapiBM25Model
from gensim.similarities import SparseMatrixSimilarity

import gradio as gr

torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", torch_device)
threads = torch.get_num_threads()
print("CPU threads:", threads)

PandasDataFrame = TypeVar('pd.core.frame.DataFrame')

embeddings = None  # global variable setup
vectorstore = None # global variable setup

full_text = "" # Define dummy source text (full text) just to enable highlight function to load

ctrans_llm = [] # Define empty list to hold CTrans LLMs for functions to run

temperature: float = 0.1
top_k: int = 3
top_p: float = 1
repetition_penalty: float = 1.05
last_n_tokens: int = 64
max_new_tokens: int = 125
#seed: int = 42
reset: bool = False
stream: bool = True
threads: int = threads
batch_size:int = 512
context_length:int = 2048
gpu_layers:int = 0
sample = False

## Highlight text constants
hlt_chunk_size = 20
hlt_strat = [" ", ".", "!", "?", ":", "\n\n", "\n", ","]
hlt_overlap = 0

## Initialise NER model ##
ner_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")

## Initialise keyword model ##
# Used to pull out keywords from chat history to add to user queries behind the scenes
kw_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

## Chat models ##
ctrans_llm = [] # Not leaded by default
#ctrans_llm = AutoModelForCausalLM.from_pretrained('TheBloke/orca_mini_3B-GGML', model_type='llama', model_file='orca-mini-3b.ggmlv3.q4_0.bin')
#ctrans_llm = AutoModelForCausalLM.from_pretrained('TheBloke/orca_mini_3B-GGML', model_type='llama', model_file='orca-mini-3b.ggmlv3.q8_0.bin')
#gpt4all_model = GPT4All(model_name= "orca-mini-3b.ggmlv3.q4_0.bin", model_path="models/") # "ggml-mpt-7b-chat.bin"

# Huggingface chat model
hf_checkpoint = 'declare-lab/flan-alpaca-large'

def create_hf_model(model_name):

    from transformers import AutoModelForSeq2SeqLM,  AutoModelForCausalLM

#    model_id = model_name
    
    if torch_device == "cuda":
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        elif "mpt" in model_name:
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto", trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    else:
        if "flan" in model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif "mpt" in model_name:    
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        else: 
            model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length = 2048)

    return model, tokenizer, torch_device

model, tokenizer, torch_device = create_hf_model(model_name = hf_checkpoint)

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

# # Prompt functions

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

    instruction_prompt_template_alpaca_quote = """### Instruction:
    Quote directly from the SOURCE below that best answers the QUESTION. Only quote full sentences in the correct order. If you cannot find an answer, start your response with "My best guess is: ".
    
    CONTENT: {summaries}
    
    QUESTION: {question}

    Response:"""

    instruction_prompt_template_orca = """
    ### System:
    You are an AI assistant that follows instruction extremely well. Help as much as you can.
    ### User:
    Answer the QUESTION using information from the following CONTENT.
    CONTENT: {summaries}
    QUESTION: {question}

    ### Response:"""

 

   
    INSTRUCTION_PROMPT=PromptTemplate(template=instruction_prompt_template_orca, input_variables=['question', 'summaries'])
    
    return INSTRUCTION_PROMPT, CONTENT_PROMPT

def adapt_q_from_chat_history(question, chat_history, extracted_memory, keyword_model=""):#keyword_model): # new_question_keywords, 
 
        chat_history_str, chat_history_first_q, chat_history_first_ans, max_chat_length = _get_chat_history(chat_history)

        if chat_history_str:
            # Keyword extraction is now done in the add_inputs_to_history function
            extracted_memory = extracted_memory#remove_q_stopwords(str(chat_history_first_q) + " " + str(chat_history_first_ans))
            
           
            new_question_kworded = str(extracted_memory) + ". " + question #+ " " + new_question_keywords
            #extracted_memory + " " + question
            
        else:
            new_question_kworded = question #new_question_keywords

        #print("Question output is: " + new_question_kworded)
            
        return new_question_kworded

def create_doc_df(docs_keep_out):
    # Extract content and metadata from 'winning' passages.
            content=[]
            meta=[]
            meta_url=[]
            page_section=[]
            score=[]

            for item in docs_keep_out:
                content.append(item[0].page_content)
                meta.append(item[0].metadata)
                meta_url.append(item[0].metadata['source'])
                page_section.append(item[0].metadata['page_section'])
                score.append(item[1])       

            # Create df from 'winning' passages

            doc_df = pd.DataFrame(list(zip(content, meta, page_section, meta_url, score)),
               columns =['page_content', 'metadata', 'page_section', 'meta_url', 'score'])

            docs_content = doc_df['page_content'].astype(str)
            doc_df['full_url'] = "https://" + doc_df['meta_url'] 

            return doc_df

def hybrid_retrieval(new_question_kworded, k_val, out_passages,
                           vec_score_cut_off, vec_weight, bm25_weight, svm_weight): # ,vectorstore, embeddings

            vectorstore=globals()["vectorstore"]
            embeddings=globals()["embeddings"]


            docs = vectorstore.similarity_search_with_score(new_question_kworded, k=k_val)

            print("Docs from similarity search:")
            print(docs)

            # Keep only documents with a certain score
            docs_len = [len(x[0].page_content) for x in docs]
            docs_scores = [x[1] for x in docs]

            # Only keep sources that are sufficiently relevant (i.e. similarity search score below threshold below)
            score_more_limit = pd.Series(docs_scores) < vec_score_cut_off
            docs_keep = list(compress(docs, score_more_limit))

            if docs_keep == []:
                docs_keep_as_doc = []
                docs_content = []
                docs_url = []
                return docs_keep_as_doc, docs_content, docs_url

            # Only keep sources that are at least 100 characters long
            length_more_limit = pd.Series(docs_len) >= 100
            docs_keep = list(compress(docs_keep, length_more_limit))

            if docs_keep == []:
                docs_keep_as_doc = []
                docs_content = []
                docs_url = []
                return docs_keep_as_doc, docs_content, docs_url

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

                return docs_keep_as_doc, docs_content, docs_url
            
            # Check for if more docs are removed than the desired output
            if out_passages > docs_keep_length: 
                out_passages = docs_keep_length
                k_val = docs_keep_length
                     
            vec_rank = [*range(1, docs_keep_length+1)]
            vec_score = [(docs_keep_length/x)*vec_weight for x in vec_rank]

            # 2nd level check on retrieved docs with BM25

            content_keep=[]
            for item in docs_keep:
                content_keep.append(item[0].page_content)

            corpus = corpus = [doc.lower().split() for doc in content_keep]
            dictionary = Dictionary(corpus)
            bm25_model = OkapiBM25Model(dictionary=dictionary)
            bm25_corpus = bm25_model[list(map(dictionary.doc2bow, corpus))]
            bm25_index = SparseMatrixSimilarity(bm25_corpus, num_docs=len(corpus), num_terms=len(dictionary),
                                   normalize_queries=False, normalize_documents=False)
            query = new_question_kworded.lower().split()
            tfidf_model = TfidfModel(dictionary=dictionary, smartirs='bnn')  # Enforce binary weighting of queries
            tfidf_query = tfidf_model[dictionary.doc2bow(query)]
            similarities = np.array(bm25_index[tfidf_query])
            #print(similarities)
            temp = similarities.argsort()
            ranks = np.arange(len(similarities))[temp.argsort()][::-1]

            # Pair each index with its corresponding value
            pairs = list(zip(ranks, docs_keep_as_doc))
            # Sort the pairs by the indices
            pairs.sort()
            # Extract the values in the new order
            bm25_result = [value for ranks, value in pairs]
            
            bm25_rank=[]
            bm25_score = []

            for vec_item in docs_keep:
                x = 0
                for bm25_item in bm25_result:
                    x = x + 1
                    if bm25_item.page_content == vec_item[0].page_content:
                        bm25_rank.append(x)
                        bm25_score.append((docs_keep_length/x)*bm25_weight)

            # 3rd level check on retrieved docs with SVM retriever
            svm_retriever = SVMRetriever.from_texts(content_keep, embeddings, k = k_val)
            svm_result = svm_retriever.get_relevant_documents(new_question_kworded)

         
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
    
    def get_docs_from_vstore(vectorstore):
        vector = vectorstore.docstore._dict
        return list(vector.items())

    def extract_details(docs_list):
        docs_list_out = [tup[1] for tup in docs_list]
        content = [doc.page_content for doc in docs_list_out]
        meta = [doc.metadata for doc in docs_list_out]
        return ''.join(content), meta[0], meta[-1]
    
    def get_parent_content_and_meta(vstore_docs, width, target):
        target_range = range(max(0, target - width), min(len(vstore_docs), target + width + 1))
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

    vstore_docs = get_docs_from_vstore(vectorstore)
    parent_vstore_meta_section = [doc.metadata['page_section'] for _, doc in vstore_docs]

    #print(docs)

    expanded_docs = []
    for doc, score in docs:
        search_section = doc.metadata['page_section']
        search_index = parent_vstore_meta_section.index(search_section) if search_section in parent_vstore_meta_section else -1
        
        content_str, meta_first, meta_last = get_parent_content_and_meta(vstore_docs, width, search_index)
        #print("Meta first:")
        #print(meta_first)
        #print("Meta last:")
        #print(meta_last)
        #print("Meta last end.")
        meta_full = merge_two_lists_of_dicts(meta_first, meta_last)

        #print(meta_full)
        
        expanded_doc = (Document(page_content=content_str[0], metadata=meta_full[0]), score)
        expanded_docs.append(expanded_doc)

    doc_df = create_doc_df(expanded_docs)  # Assuming you've defined the 'create_doc_df' function elsewhere

    return expanded_docs, doc_df

def create_final_prompt(inputs: Dict[str, str], instruction_prompt, content_prompt, extracted_memory): # , 
        
        question =  inputs["question"]
        chat_history = inputs["chat_history"]
        

        new_question_kworded = adapt_q_from_chat_history(question, chat_history, extracted_memory) # new_question_keywords, 
        

        #print("The question passed to the vector search is:")
        #print(new_question_kworded)
        
        #docs_keep_as_doc, docs_content, docs_url = find_relevant_passages(new_question_kworded, k_val = 5, out_passages = 3,
        #                                                                  vec_score_cut_off = 1.3, vec_weight = 1, tfidf_weight = 0.5, svm_weight = 1)

        docs_keep_as_doc, doc_df, docs_keep_out = hybrid_retrieval(new_question_kworded, k_val = 5, out_passages = 2,
                                                                          vec_score_cut_off = 1, vec_weight = 1, bm25_weight = 1, svm_weight = 1)#,
                                                                          #vectorstore=globals()["vectorstore"], embeddings=globals()["embeddings"])
        
        # Expand the found passages to the neighbouring context
        docs_keep_as_doc, doc_df = get_expanded_passages(vectorstore, docs_keep_out, width=1)

        if docs_keep_as_doc == []:
            {"answer": "I'm sorry, I couldn't find a relevant answer to this question.", "sources":"I'm sorry, I couldn't find a relevant source for this question."}
        
        #new_inputs = inputs.copy()
        #new_inputs["question"] = new_question
        #new_inputs["chat_history"] = chat_history_str
        
        #print(docs_url)
        #print(doc_df['metadata'])

        # Build up sources content to add to user display

        doc_df['meta_clean'] = [f"<b>{'  '.join(f'{k}: {v}' for k, v in d.items() if k != 'page_section')}</b>" for d in doc_df['metadata']]
        doc_df['content_meta'] = doc_df['meta_clean'].astype(str) + ".<br><br>" + doc_df['page_content'].astype(str)

        modified_page_content = [f" SOURCE {i+1} - {word}" for i, word in enumerate(doc_df['page_content'])]
        docs_content_string = ''.join(modified_page_content)

        #docs_content_string = '<br><br>\n\n SOURCE '.join(doc_df['page_content'])#.replace("  "," ")#.strip()
        sources_docs_content_string = '<br><br>'.join(doc_df['content_meta'])#.replace("  "," ")#.strip()
        #sources_docs_content_tup = [(sources_docs_content,None)]
        #print("The draft instruction prompt is:")
        #print(instruction_prompt)
        
        instruction_prompt_out = instruction_prompt.format(question=new_question_kworded, summaries=docs_content_string)
        #print("The final instruction prompt:")
        #print(instruction_prompt_out)
        
                
        return instruction_prompt_out, sources_docs_content_string, new_question_kworded

def get_history_sources_final_input_prompt(user_input, history, extracted_memory):#):
    
    #if chain_agent is None:
    #    history.append((user_input, "Please click the button to submit the Huggingface API key before using the chatbot (top right)"))
    #    return history, history, "", ""
    print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
    print("User input: " + user_input)
    
    history = history or []
    

    
    # Create instruction prompt
    instruction_prompt, content_prompt = create_prompt_templates()
    instruction_prompt_out, docs_content_string, new_question_kworded =\
                create_final_prompt({"question": user_input, "chat_history": history}, #vectorstore,
                                    instruction_prompt, content_prompt, extracted_memory)
    
  
    history.append(user_input)
    
    print("Output history is:")
    print(history)

    #print("The output prompt is:")
    #print(instruction_prompt_out)
    
    return history, docs_content_string, instruction_prompt_out

def highlight_found_text_single(search_text:str, full_text:str, hlt_chunk_size:int=hlt_chunk_size, hlt_strat:List=hlt_strat, hlt_overlap:int=hlt_overlap) -> str:
    """
    Highlights occurrences of search_text within full_text.
    
    Parameters:
    - search_text (str): The text to be searched for within full_text.
    - full_text (str): The text within which search_text occurrences will be highlighted.
    
    Returns:
    - str: A string with occurrences of search_text highlighted.
    
    Example:
    >>> highlight_found_text("world", "Hello, world! This is a test. Another world awaits.")
    'Hello, <mark style="color:black;">world</mark>! This is a test. Another world awaits.'
    """

    def extract_text_from_input(text,i=0):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()#.replace("\r", " ").replace("\n", " ")
        elif isinstance(text, list):
            return text[i][0].replace("  ", " ").strip()#.replace("\r", " ").replace("\n", " ")
        else:
            return ""
        
    def extract_search_text_from_input(text):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()#.replace("\r", " ").replace("\n", " ").replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[-1][1].replace("  ", " ").strip()#.replace("\r", " ").replace("\n", " ").replace("  ", " ").strip()
        else:
            return ""
            
    full_text = extract_text_from_input(full_text)
    search_text = extract_search_text_from_input(search_text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hlt_chunk_size,
        separators=hlt_strat,
        chunk_overlap=hlt_overlap,
    )
    sections = text_splitter.split_text(search_text)

    #print(sections)

    found_positions = {}
    for x in sections:
        text_start_pos = full_text.find(x)
        
        if text_start_pos != -1:
            found_positions[text_start_pos] = text_start_pos + len(x)

    # Combine overlapping or adjacent positions
    sorted_starts = sorted(found_positions.keys())
    combined_positions = []
    if sorted_starts:
        current_start, current_end = sorted_starts[0], found_positions[sorted_starts[0]]
        for start in sorted_starts[1:]:
            if start <= (current_end + 1):
                current_end = max(current_end, found_positions[start])
            else:
                combined_positions.append((current_start, current_end))
                current_start, current_end = start, found_positions[start]
        combined_positions.append((current_start, current_end))

    # Construct pos_tokens
    pos_tokens = []
    prev_end = 0
    for start, end in combined_positions:
        pos_tokens.append(full_text[prev_end:start]) # ((full_text[prev_end:start], None))
        pos_tokens.append('<mark style="color:black;">' + full_text[start:end] + '</mark>')# ("<mark>" + full_text[start:end] + "</mark>",'found')
        prev_end = end
    pos_tokens.append(full_text[prev_end:])

    return "".join(pos_tokens)

def highlight_found_text(search_text: str, full_text: str, hlt_chunk_size:int=hlt_chunk_size, hlt_strat:List=hlt_strat, hlt_overlap:int=hlt_overlap) -> str:
    """
    Highlights occurrences of search_text within full_text.
    
    Parameters:
    - search_text (str): The text to be searched for within full_text.
    - full_text (str): The text within which search_text occurrences will be highlighted.
    
    Returns:
    - str: A string with occurrences of search_text highlighted.
    
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

    def extract_search_text_from_input(text):
        if isinstance(text, str):
            return text.replace("  ", " ").strip()
        elif isinstance(text, list):
            return text[-1][1].replace("  ", " ").strip()
        else:
            return ""

    full_text = extract_text_from_input(full_text)
    search_text = extract_search_text_from_input(search_text)



    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=hlt_chunk_size,
        separators=hlt_strat,
        chunk_overlap=hlt_overlap,
    )
    sections = text_splitter.split_text(search_text)

    found_positions = {}
    for x in sections:
        text_start_pos = 0
        while text_start_pos != -1:
            text_start_pos = full_text.find(x, text_start_pos)
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
        pos_tokens.append(full_text[prev_end:start])
        pos_tokens.append('<mark style="color:black;">' + full_text[start:end] + '</mark>')
        prev_end = end
    pos_tokens.append(full_text[prev_end:])

    return "".join(pos_tokens)

# # Chat functions
def produce_streaming_answer_chatbot_gpt4all(history, full_prompt): 
    
    print("The question is: ")
    print(full_prompt)

    # Pull the generated text from the streamer, and update the model output. 
    history[-1][1] = ""
    for new_text in gpt4all_model.generate(full_prompt, max_tokens=2000, streaming=True):
        if new_text == None: new_text = ""
        history[-1][1] += new_text
        yield history
          
def produce_streaming_answer_chatbot_hf(history, full_prompt): 
    
    #print("The question is: ")
    #print(full_prompt)
    
    # Get the model and tokenizer, and tokenize the user text.
    model_inputs = tokenizer(text=full_prompt, return_tensors="pt").to(torch_device)

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, timeout=60., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=sample,
        repetition_penalty=1.3,
        top_p=top_p,
        temperature=temperature,
        top_k=top_k
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    import time
    start = time.time()
    NUM_TOKENS=0
    print('-'*4+'Start Generation'+'-'*4)

    history[-1][1] = ""
    for new_text in streamer:
        if new_text == None: new_text = ""
        history[-1][1] += new_text
        NUM_TOKENS+=1
        yield history
        
    time_generate = time.time() - start
    print('\n')
    print('-'*4+'End Generation'+'-'*4)
    print(f'Num of generated tokens: {NUM_TOKENS}')
    print(f'Time for complete generation: {time_generate}s')
    print(f'Tokens per secound: {NUM_TOKENS/time_generate}')
    print(f'Time per token: {(time_generate/NUM_TOKENS)*1000}ms')

def produce_streaming_answer_chatbot_ctrans(history, full_prompt): 
    
    print("The question is: ")
    print(full_prompt)

    #tokens = ctrans_llm.tokenize(full_prompt)

    #import psutil
    #from loguru import logger

    #_ = [elm for elm in full_prompt.splitlines() if elm.strip()]
    #stop_string = [elm.split(":")[0] + ":" for elm in _][-2]
    #print(stop_string)

    #logger.debug(f"{stop_string=} not used")

    #_ = psutil.cpu_count(logical=False) - 1
    #cpu_count: int = int(_) if _ else 1
    #logger.debug(f"{cpu_count=}")

    # Pull the generated text from the streamer, and update the model output.
    config = GenerationConfig(reset=True)
    history[-1][1] = ""
    for new_text in ctrans_generate(prompt=full_prompt, config=config):
        if new_text == None: new_text = ""
        history[-1][1] += new_text
        yield history

@dataclass
class GenerationConfig:
    temperature: float = temperature
    top_k: int = top_k
    top_p: float = top_p
    repetition_penalty: float = repetition_penalty
    last_n_tokens: int = last_n_tokens
    max_new_tokens: int = max_new_tokens
    #seed: int = 42
    reset: bool = reset
    stream: bool = stream
    threads: int = threads
    batch_size:int = batch_size
    #context_length:int = context_length
    #gpu_layers:int = gpu_layers
    #stop: list[str] = field(default_factory=lambda: [stop_string])

def ctrans_generate(
    prompt: str,
    llm=ctrans_llm,
    config: GenerationConfig = GenerationConfig(),
):
    """Run model inference, will return a Generator if streaming is true."""

    return llm(
        prompt,
        **asdict(config),
    )

def turn_off_interactivity(user_message, history):
        return gr.update(value="", interactive=False), history + [[user_message, None]]

# # Chat history functions

def clear_chat(chat_history_state, sources, chat_message, current_topic):
    chat_history_state = []
    sources = ''
    chat_message = ''
    current_topic = ''

    return chat_history_state, sources, chat_message, current_topic

def _get_chat_history(chat_history: List[Tuple[str, str]], max_chat_length:int = 20): # Limit to last x interactions only

    if not chat_history:
        chat_history = []

    if len(chat_history) > max_chat_length:
        chat_history = chat_history[-max_chat_length:]
        
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

    return conversation, first_q, first_ans, max_chat_length

def add_inputs_answer_to_history(user_message, history, current_topic):
    
    #history.append((user_message, [-1]))

    chat_history_str, chat_history_first_q, chat_history_first_ans, max_chat_length = _get_chat_history(history)


    # Only get the keywords for the first question and response, or do it every time if over 'max_chat_length' responses in the conversation
    if (len(history) == 1) | (len(history) > max_chat_length):
        
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

    keywords_text = keybert.KeyBERT(model=kw_model).extract_keywords(lemmatised_text, stop_words='english', top_n=n, 
                                                   keyphrase_ngram_range=(1, 1))
    keywords_list = [item[0] for item in keywords_text]

    return keywords_list

