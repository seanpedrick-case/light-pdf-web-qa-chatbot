import os
import google.generativeai as ai
import pandas as pd
import gradio as gr
import markdown
import time
import boto3
import json
import string
import re
from gradio import Progress
from typing import List, Tuple
from io import StringIO

from tools.prompts import prompt1, prompt2, prompt3, system_prompt, summarise_system_prompt, summarise_prompt
from tools.helper_functions import output_folder, detect_file_type, get_file_path_end, read_file, get_or_create_env_var
from tools.config import GEMINI_MODELS

# ResponseObject class for AWS Bedrock calls
class ResponseObject:
        def __init__(self, text, usage_metadata):
            self.text = text
            self.usage_metadata = usage_metadata

max_tokens = 4096


AWS_DEFAULT_REGION = get_or_create_env_var('AWS_DEFAULT_REGION', 'eu-west-2')
print(f'The value of AWS_DEFAULT_REGION is {AWS_DEFAULT_REGION}')

bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_DEFAULT_REGION)

def normalise_string(text):
    # Replace two or more dashes with a single dash
    text = re.sub(r'-{2,}', '-', text)
    
    # Replace two or more spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)
    
    return text

def load_in_file(file_path: str, colname:str):
    """
    Loads in a tabular data file and returns data and file name.

    Parameters:
    - file_path (str): The path to the file to be processed.
    """
    file_type = detect_file_type(file_path)
    print("File type is:", file_type)

    out_file_part = get_file_path_end(file_path)
    file_data = read_file(file_path)

    file_data[colname].fillna("", inplace=True)

    file_data[colname] = file_data[colname].astype(str).str.replace("\bnan\b", "", regex=True)  
    

    print(file_data[colname])

    return file_data, out_file_part

def load_in_data_file(file_paths:List[str], in_colnames:List[str], batch_size:int=50, ):
    '''Load in data table, work out how many batches needed.'''

    try:
        file_data, file_name = load_in_file(file_paths[0], colname=in_colnames)
        num_batches = (len(file_data) // batch_size) + 1

    except Exception as e:
        print(e)
        file_data = pd.DataFrame()
        file_name = ""
        num_batches = 1  
    
    return file_data, file_name, num_batches

def data_file_to_markdown_table(file_data:pd.DataFrame, file_name:str, chosen_cols: List[str], output_folder: str, batch_number: int, batch_size: int) -> Tuple[str, str, str]:
    """
    Processes a file by simplifying its content based on chosen columns and saves the result to a specified output folder.

    Parameters:
    - file_data (pd.DataFrame): Tabular data file with responses.
    - file_name (str): File name with extension.
    - chosen_cols (List[str]): A list of column names to include in the simplified file.
    - output_folder (str): The directory where the simplified file will be saved.
    - batch_number (int): The current batch number for processing.
    - batch_size (int): The number of rows to process in each batch.

    Returns:
    - Tuple[str, str, str]: A tuple containing the path to the simplified CSV file, the simplified markdown table as a string, and the file path end (used for naming the output file).
    """

    #print("\nfile_data_in_markdown func:", file_data)
    #print("\nBatch size in markdown func:", str(batch_size))
    
    normalised_simple_markdown_table = ""
    simplified_csv_table_path = ""

    # Simplify table to just responses column and the Response reference number
    simple_file = file_data[[chosen_cols]].reset_index(names="Reference")
    simple_file["Reference"] = simple_file["Reference"].astype(int) + 1
    simple_file = simple_file.rename(columns={chosen_cols: "Response"})
    simple_file["Response"] = simple_file["Response"].str.strip()
    file_len = len(simple_file["Reference"])
   

     # Subset the data for the current batch
    start_row = batch_number * batch_size
    if start_row > file_len + 1:
        print("Start row greater than file row length")
        return simplified_csv_table_path, normalised_simple_markdown_table, file_name

    if (start_row + batch_size) <= file_len + 1:
        end_row = start_row + batch_size
    else:
        end_row = file_len + 1

    simple_file = simple_file[start_row:end_row]  # Select the current batch

    # Remove problematic characters including ASCII and various quote marks
        # Remove problematic characters including control characters, special characters, and excessive leading/trailing whitespace
    simple_file["Response"] = simple_file["Response"].str.replace(r'[\x00-\x1F\x7F]|["“”‘’<>]|\\', '', regex=True)  # Remove control and special characters
    simple_file["Response"] = simple_file["Response"].str.strip()  # Remove leading and trailing whitespace
    simple_file["Response"] = simple_file["Response"].str.replace(r'\s+', ' ', regex=True)  # Replace multiple spaces with a single space

    # Remove blank and extremely short responses
    simple_file = simple_file.loc[~(simple_file["Response"].isnull()) & ~(simple_file["Response"].str.len() < 5), :]

    simplified_csv_table_path = output_folder + 'simple_markdown_table_' + file_name + '_row_' + str(start_row) + '_to_' + str(end_row) + '.csv'
    simple_file.to_csv(simplified_csv_table_path, index=None)

    simple_markdown_table = simple_file.to_markdown(index=None)

    normalised_simple_markdown_table = normalise_string(simple_markdown_table)

    return simplified_csv_table_path, normalised_simple_markdown_table, file_name, start_row, end_row

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

    #model = ai.GenerativeModel.from_cached_content(cached_content=cache, generation_config=config)
    model = ai.GenerativeModel(model_name='models/' + model_choice, system_instruction=system_prompt, generation_config=config)
    
    return model, config

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
        role = entry['role'].capitalize()  # Assuming the history is stored with 'role' and 'parts'
        message = ' '.join(entry['parts'])  # Combining all parts of the message
        full_prompt += f"{role}: {message}\n"
    
    # Adding the new user prompt
    full_prompt += f"\nUser: {prompt}"

    # Print the full prompt for debugging purposes
    #print("full_prompt:", full_prompt)

    # Generate the model's response
    if model_choice in GEMINI_MODELS:
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
    else:
        try:
            print("Calling AWS Claude model")
            response = call_aws_claude(prompt, system_prompt, temperature, max_tokens, model_choice)
        except Exception as e:
            # If fails, try again after 10 seconds in case there is a throttle limit
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

    # Update the conversation history with the new prompt and response
    conversation_history.append({'role': 'user', 'parts': [prompt]})
    conversation_history.append({'role': 'assistant', 'parts': [response.text]})
    
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
    for prompt in prompts:

        response, conversation_history = send_request(prompt, conversation_history, model=model, config=config, model_choice=model_choice, system_prompt=system_prompt_with_table, temperature=temperature)
        
        #print(response.text)
        print(response.usage_metadata)
        responses.append(response)

        # Create conversation txt object
        whole_conversation.append(prompt)
        whole_conversation.append(response.text)

        # Create conversation metadata
        if master == False:
            whole_conversation_metadata.append(f"Query batch {batch_no} prompt {len(responses)} metadata:")
        else:
            whole_conversation_metadata.append(f"Query summary metadata:")

        whole_conversation_metadata.append(str(response.usage_metadata))

    return responses, conversation_history, whole_conversation, whole_conversation_metadata

def replace_punctuation_with_underscore(input_string):
    # Create a translation table where each punctuation character maps to '_'
    translation_table = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    
    # Translate the input string using the translation table
    return input_string.translate(translation_table)

def clean_markdown_table(text: str):
    lines = text.splitlines()

    # Remove any empty rows or rows with only pipes
    cleaned_lines = [line for line in lines if not re.match(r'^\s*\|?\s*\|?\s*$', line)]

    # Merge lines that belong to the same row (i.e., don't start with |)
    merged_lines = []
    buffer = ""
    
    for line in cleaned_lines:
        if line.lstrip().startswith('|'):  # If line starts with |, it's a new row
            if buffer:
                merged_lines.append(buffer)  # Append the buffered content
            buffer = line  # Start a new buffer with this row
        else:
            # Continuation of the previous row
            buffer += ' ' + line.strip()  # Add content to the current buffer

    # Don't forget to append the last buffer
    if buffer:
        merged_lines.append(buffer)

    # Ensure consistent number of pipes in each row based on the header
    header_pipes = merged_lines[0].count('|')  # Use the first row to count number of pipes
    result = []

    for line in merged_lines:
        # Strip excessive whitespace around pipes
        line = re.sub(r'\s*\|\s*', '|', line.strip())

        # Replace numbers between pipes with commas and a space
        line = re.sub(r'(?<=\|)(\s*\d+)(,\s*\d+)+(?=\|)', lambda m: ', '.join(m.group(0).split(',')), line)

        # Replace groups of numbers separated by spaces with commas and a space
        line = re.sub(r'(?<=\|)(\s*\d+)(\s+\d+)+(?=\|)', lambda m: ', '.join(m.group(0).split()), line)

        # Fix inconsistent number of pipes by adjusting them to match the header
        pipe_count = line.count('|')
        if pipe_count < header_pipes:
            line += '|' * (header_pipes - pipe_count)  # Add missing pipes
        elif pipe_count > header_pipes:
            # If too many pipes, split line and keep the first `header_pipes` columns
            columns = line.split('|')[:header_pipes + 1]  # +1 to keep last pipe at the end
            line = '|'.join(columns)

        result.append(line)

    # Join lines back into the cleaned markdown text
    cleaned_text = '\n'.join(result)

    return cleaned_text

def write_llm_output_and_logs(responses: List[ResponseObject], whole_conversation: List[str], whole_conversation_metadata: List[str], out_file_part: str, latest_batch_completed: int, start_row:int, end_row:int, model_choice_clean: str, temperature: float, log_files_output_paths: List[str], existing_reference_df:pd.DataFrame, existing_topics_df:pd.DataFrame, first_run: bool = False) -> None:
    """
    Writes the output of the large language model requests and logs to files.

    Parameters:
    - responses (List[ResponseObject]): A list of ResponseObject instances containing the text and usage metadata of the responses.
    - whole_conversation (List[str]): A list of strings representing the complete conversation including prompts and responses.
    - whole_conversation_metadata (List[str]): A list of strings representing metadata about the whole conversation.
    - out_file_part (str): The base part of the output file name.
    - latest_batch_completed (int): The index of the current batch.
    - start_row (int): Start row of the current batch.
    - end_row (int): End row of the current batch.
    - model_choice_clean (str): The cleaned model choice string.
    - temperature (float): The temperature parameter used in the model.
    - log_files_output_paths (List[str]): A list of paths to the log files.
    - existing_reference_df (pd.DataFrame): The existing reference dataframe mapping response numbers to topics.
    - existing_topics_df (pd.DataFrame): The existing unique topics dataframe
    - first_run (bool): A boolean indicating if this is the first run through this function in this process. Defaults to False.
    """
    unique_topics_df_out_path = []
    topic_table_out_path = "topic_table_error.csv"
    reference_table_out_path = "reference_table_error.csv"
    unique_topics_df_out_path = "unique_topic_table_error.csv"
    topic_with_response_df = pd.DataFrame()
    markdown_table = ""
    out_reference_df = pd.DataFrame()
    out_unique_topics_df = pd.DataFrame()
    batch_out_file_part = "error"


    # If there was an error in parsing, return boolean saying error
    is_error = False

    # Convert conversation to string and add to log outputs
    whole_conversation_str = '\n'.join(whole_conversation)
    whole_conversation_metadata_str = '\n'.join(whole_conversation_metadata)

    start_row_reported = start_row + 1

    # Save outputs for each batch. If master file created, label file as master
    if first_run == True:
        batch_out_file_part = f"{out_file_part}_batch_{latest_batch_completed + 1}"
        batch_part = f"Rows {start_row_reported} to {end_row}: "
    else:
        batch_out_file_part = f"{out_file_part}_combined_batch_{latest_batch_completed + 1}"
        batch_part = f"Rows {start_row_reported} to {end_row}: "

    whole_conversation_path = output_folder + batch_out_file_part + "_full_conversation_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"
    whole_conversation_path_meta = output_folder + batch_out_file_part + "_metadata_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

    # print("whole_conversation:", whole_conversation_str)

    with open(whole_conversation_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_str)

    with open(whole_conversation_path_meta, "w", encoding='utf-8', errors='replace') as f:
        f.write(whole_conversation_metadata_str)

    log_files_output_paths.append(whole_conversation_path)
    log_files_output_paths.append(whole_conversation_path_meta)

    # Convert output table to markdown and then to a pandas dataframe to csv
    # try:
    cleaned_response = clean_markdown_table(responses[-1].text)
    
    markdown_table = markdown.markdown(cleaned_response, extensions=['tables'])

    #print("markdown_table:", markdown_table)

    # Remove <p> tags and make sure it has a valid HTML structure
    html_table = re.sub(r'<p>(.*?)</p>', r'\1', markdown_table)
    html_table = html_table.replace('<p>', '').replace('</p>', '').strip()

    # Now ensure that the HTML structure is correct
    if "<table>" not in html_table:
        html_table = f"""
        <table>
            {html_table}
        </table>
        """

    # print("Markdown table as HTML:", html_table)

    html_buffer = StringIO(html_table)
    #print("html_buffer:", html_buffer)

    try:
        topic_with_response_df = pd.read_html(html_buffer)[0]  # Assuming the first table in the HTML is the one you want
    except Exception as e:
        print("Error when trying to parse table:", e)
        is_error = True
        return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_out_file_part, is_error


    # Rename columns to ensure consistent use of data frames later in code
    topic_with_response_df.columns = ["General Topic", "Subtopic", "Sentiment", "Summary", "Response References"]

    # Fill in NA rows with values from above (topics seem to be included only on one row):
    topic_with_response_df = topic_with_response_df.ffill()

    topic_table_out_path = output_folder + batch_out_file_part + "_topic_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

    # Table to map references to topics
    reference_data = []

    # Iterate through each row in the original DataFrame
    for index, row in topic_with_response_df.iterrows():
        references = re.split(r',\s*|\s+', str(row.iloc[4]))  # Split the reference numbers
        topic = row.iloc[0]
        subtopic = row.iloc[1]
        sentiment = row.iloc[2]
        summary = row.iloc[3]

        summary = batch_part + summary

        # Create a new entry for each reference number
        for ref in references:
            reference_data.append({
                'Response References': ref,
                'General Topic': topic,
                'Subtopic': subtopic,
                'Sentiment': sentiment,
                'Summary': summary,
                "Start row of group": start_row_reported
            })

    # Create a new DataFrame from the reference data
    new_reference_df = pd.DataFrame(reference_data)
    
    # Append on old reference data
    out_reference_df = pd.concat([new_reference_df, existing_reference_df])

    # Remove duplicate Response references for the same topic
    out_reference_df.drop_duplicates(["Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)

    out_reference_df.sort_values(["Start row of group", "Response References", "General Topic", "Subtopic", "Sentiment"], inplace=True)


    reference_counts = out_reference_df.groupby(["General Topic", "Subtopic", "Sentiment"]).agg({
    'Response References': 'size',  # Count the number of references
    'Summary': lambda x: '<br>'.join(
        sorted(set(x), key=lambda summary: out_reference_df.loc[out_reference_df['Summary'] == summary, 'Start row of group'].min())
    )
    }).reset_index()

    # Save the new DataFrame to CSV
    reference_table_out_path = output_folder + batch_out_file_part + "_reference_table_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"
    
    # Table of all unique topics with descriptions
    new_unique_topics_df = topic_with_response_df[["General Topic", "Subtopic", "Sentiment"]] # , "Summary"
    
    # Join existing and new unique topics
    out_unique_topics_df = pd.concat([new_unique_topics_df, existing_topics_df]).drop_duplicates(["Subtopic"]).drop(["Response References", "Summary"], axis = 1, errors="ignore")        

    # Join the counts to existing_unique_topics_df
    out_unique_topics_df = out_unique_topics_df.merge(reference_counts, how='left', on=["General Topic", "Subtopic", "Sentiment"]).sort_values("Response References", ascending=False)

    unique_topics_df_out_path = output_folder + batch_out_file_part + "_unique_topics_" + model_choice_clean + "_temp_" + str(temperature) + ".csv"

    # except Exception as e:
    #     print("Error in write_llm_output_and_logs:")
    #     print(e)

    return topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_with_response_df, markdown_table, out_reference_df, out_unique_topics_df, batch_out_file_part, is_error

def llm_query(file_data:pd.DataFrame, existing_topics_w_references_table:pd.DataFrame, existing_reference_df:pd.DataFrame, existing_unique_topics_df:pd.DataFrame, display_table:str, file_name:str, num_batches:int, in_api_key:str, temperature:float, chosen_cols:List[str], model_choice:str, candidate_topics: List=[],latest_batch_completed:int=0, out_message:List=[], out_file_paths:List = [], log_files_output_paths:List = [], first_loop_state:bool=False, whole_conversation_metadata_str:str="", prompt1:str=prompt1, prompt2:str=prompt2, prompt3:str=prompt3, system_prompt:str=system_prompt, summarise_system_prompt:str=summarise_system_prompt, summarise_prompt:str=summarise_prompt, number_of_requests:int=1, batch_size:int=50, max_tokens:int=max_tokens, progress=Progress(track_tqdm=True)):

    '''
    Query an LLM (Gemini or AWS Anthropic-based) with up to three prompts about a table of open text data. Up to 'batch_size' rows will be queried at a time.

    Parameters:
    - file_data (pd.DataFrame): Pandas dataframe containing the consultation response data.
    - existing_topics_w_references_table (pd.DataFrame): Pandas dataframe containing the latest master topic table that has been iterated through batches.
    - existing_reference_df (pd.DataFrame): Pandas dataframe containing the list of Response reference numbers alongside the derived topics and subtopics.
    - existing_unique_topics_df (pd.DataFrame): Pandas dataframe containing the unique list of topics, subtopics, sentiment and summaries until this point.
    - display_table (str): Table for display in markdown format.
    - file_name (str): File name of the data file.
    - num_batches (int): Number of batches required to go through all the response rows.
    - in_api_key (str): The API key for authentication.
    - temperature (float): The temperature parameter for the model.
    - chosen_cols (List[str]): A list of chosen columns to process.
    - candidate_topics (List): A list of existing candidate topics submitted by the user.
    - model_choice (str): The choice of model to use.
    - latest_batch_completed (int): The index of the latest file completed.
    - out_message (list): A list to store output messages.
    - out_file_paths (list): A list to store output file paths.
    - log_files_output_paths (list): A list to store log file output paths.
    - first_loop_state (bool): A flag indicating the first loop state.
    - whole_conversation_metadata_str (str): A string to store whole conversation metadata.
    - prompt1 (str): The first prompt for the model.
    - prompt2 (str): The second prompt for the model.
    - prompt3 (str): The third prompt for the model.
    - system_prompt (str): The system prompt for the model.
    - summarise_system_prompt (str): The system prompt for the summary part of the model.
    - summarise_prompt (str): The prompt for the model summary.
    - number of requests (int): The number of prompts to send to the model.
    - batch_size (int): The number of data rows to consider in each request.
    - max_tokens (int): The maximum number of tokens for the model.
    - progress (Progress): A progress tracker.
    '''

    tic = time.perf_counter()
    model = ""
    config = ""
    final_time = 0.0
    whole_conversation_metadata = []
    all_topic_tables_df = []
    all_markdown_topic_tables = []
    is_error = False

    # Reset output files on each run:
    # out_file_paths = []

    model_choice_clean = replace_punctuation_with_underscore(model_choice)

    # If this is the first time around, set variables to 0/blank
    if first_loop_state==True:
        latest_batch_completed = 0
        out_message = []
        out_file_paths = []

    print("latest_batch_completed:", str(latest_batch_completed))

    if num_batches > 0:
        progress_measure = round(latest_batch_completed / num_batches, 1)
        progress(progress_measure, desc="Querying large language model")
    else:
        progress(0.1, desc="Querying large language model")

    # Load file
    # If out message or out_file_paths are blank, change to a list so it can be appended to
    if isinstance(out_message, str):
        out_message = [out_message]

    if not out_file_paths:
        out_file_paths = []

    # Check if files and text exist
    if file_data.empty:
        out_message = "Please enter text or a file to redact."
        return out_message, existing_topics_w_references_table, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_message
    
    if model_choice == "anthropic.claude-3-sonnet-20240229-v1:0" and file_data.shape[1] > 300:
        out_message = "Your data has more than 300 rows, using the Sonnet model will be too expensive. Please choose the Haiku model instead."
        return out_message, existing_topics_w_references_table, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, out_message
        
    # If we have already redacted the last file, return the input out_message and file list to the relevant components
    if latest_batch_completed >= num_batches:
        print("Last batch reached, returning batch:", str(latest_batch_completed))
        # Set to a very high number so as not to mess with subsequent file processing by the user
        latest_batch_completed = 999

        toc = time.perf_counter()
        final_time = toc - tic
        #out_time = f"in {final_time} seconds."
        #print(out_time)  

        final_out_message = '\n'.join(out_message)
        return display_table, existing_topics_w_references_table, existing_unique_topics_df, existing_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_out_message
    
    #for latest_batch_completed in range(num_batches):
    reported_batch_no = latest_batch_completed + 1  
    print("Running query batch", str(reported_batch_no))

    # Call the function to prepare the input table
    simplified_csv_table_path, normalised_simple_markdown_table, out_file_part, start_row, end_row = data_file_to_markdown_table(file_data, file_name, chosen_cols, output_folder, latest_batch_completed, batch_size)
    log_files_output_paths.append(simplified_csv_table_path)


    # Conversation history
    conversation_history = []


    # If this is the second batch, the master table will refer back to the current master table when assigning topics to the new table. Also runs if there is an existing list of topics supplied by the user
    if latest_batch_completed >= 1 or candidate_topics:

        #print("normalised_simple_markdown_table:", normalised_simple_markdown_table)

        # Prepare Gemini models before query       
        if model_choice in GEMINI_MODELS:
            print("Using Gemini model:", model_choice)
            model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=summarise_system_prompt, max_tokens=max_tokens)
        else:
            print("Using AWS Bedrock model:", model_choice)

        # Merge duplicate topics together to create a big merged summary table
        #all_topic_tables_df_merged = existing_topics_w_references_table#pd.concat(all_topic_tables_df)

        # Group by the first three columns and concatenate the fourth and fifth columns
        # all_topic_tables_df_merged = existing_topics_w_references_table.groupby(["General Topic", "Subtopic", "Sentiment"], as_index=False).agg({
        #     "Summary": '\n'.join,   # Concatenate the fourth column
        #     "Response References": ', '.join  # Concatenate the fifth column
        # })
        # all_topic_tables_df_merged["Response References"] = ""
        #all_topic_tables_df_merged["Summary"] = ""
        #all_topic_tables_str = all_topic_tables_df_merged.to_markdown(index=None)

        if candidate_topics:
            # 'Zero shot topics' are those supplied by the user
            zero_shot_topics = read_file(candidate_topics.name)
            zero_shot_topics_series = zero_shot_topics.iloc[:, 0]
            # Max 150 topics allowed
            if len(zero_shot_topics_series) > 120:
                print("Maximum 120 topics allowed to fit within large language model context limits.")
                zero_shot_topics_series = zero_shot_topics_series.iloc[:120]

            zero_shot_topics_list = list(zero_shot_topics_series)

            print("Zero shot topics are:", zero_shot_topics_list)

        #all_topic_tables_df_merged = existing_unique_topics_df
        existing_unique_topics_df["Response References"] = ""

        
  

        # Create the most up to date list of topics and subtopics.
        # If there are candidate topics, but the existing_unique_topics_df hasn't yet been constructed, then create.
        if candidate_topics and existing_unique_topics_df.empty:
            existing_unique_topics_df = pd.DataFrame(data={'General Topic':'', 'Subtopic':zero_shot_topics_list, 'Sentiment':''})

        # This part concatenates all zero shot and new topics together, so that for the next prompt the LLM will have the full list available
        elif candidate_topics and not existing_unique_topics_df.empty:
            zero_shot_topics_df = pd.DataFrame(data={'General Topic':'', 'Subtopic':zero_shot_topics_list, 'Sentiment':''})
            existing_unique_topics_df = pd.concat([existing_unique_topics_df, zero_shot_topics_df]).drop_duplicates("Subtopic")

            #print("Full topics list with zero shot_dropped:", existing_unique_topics_df)

            existing_unique_topics_df.to_csv(output_folder + "Existing topics with zero shot dropped.csv")


        unique_topics_markdown = existing_unique_topics_df[["General Topic", "Subtopic", "Sentiment"]].drop_duplicates(["General Topic", "Subtopic", "Sentiment"]).to_markdown(index=False)
       
        existing_unique_topics_df.to_csv(output_folder + f"{out_file_part}_master_all_topic_tables_df_merged_" + model_choice_clean + "_temp_" + str(temperature) + "_batch_" + str(latest_batch_completed) + ".csv")

        # Format the summary prompt with the response table and topics
        formatted_summary_prompt = summarise_prompt.format(response_table=normalised_simple_markdown_table, topics=unique_topics_markdown)

        # Define the output file path for the formatted prompt
        formatted_prompt_output_path = output_folder + out_file_part + "_full_prompt_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

        # Write the formatted prompt to the specified file
        try:
            with open(formatted_prompt_output_path, "w", encoding='utf-8', errors='replace') as f:
                f.write(formatted_summary_prompt)
        except Exception as e:
            print(f"Error writing prompt to file {formatted_prompt_output_path}: {e}")

        summary_prompt_list = [formatted_summary_prompt]

        summary_conversation_history = []
        summary_whole_conversation = []

        # Process requests to large language model
        master_summary_response, summary_conversation_history, whole_summary_conversation, whole_conversation_metadata = process_requests(summary_prompt_list, summarise_system_prompt, summary_conversation_history, summary_whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no, master = True)

        new_topic_table_out_path, new_reference_table_out_path, new_unique_topics_df_out_path, new_topic_df, new_markdown_table, new_reference_df, new_unique_topics_df, master_batch_out_file_part, is_error =  write_llm_output_and_logs(master_summary_response, whole_summary_conversation, whole_conversation_metadata, out_file_part, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, first_run=False)

        # If error in table parsing, leave function
        if is_error == True:
            final_message_out = "Could not complete summary, error in LLM output."
            display_table, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_message_out

        # Write outputs to csv
        ## Topics with references
        new_topic_df.to_csv(new_topic_table_out_path, index=None)
        log_files_output_paths.append(new_topic_table_out_path)

        ## Reference table mapping response numbers to topics
        new_reference_df.to_csv(new_reference_table_out_path, index=None)
        log_files_output_paths.append(new_reference_table_out_path)

        ## Unique topic list
        new_unique_topics_df.to_csv(new_unique_topics_df_out_path, index=None)
        out_file_paths.append(new_unique_topics_df_out_path)

        all_topic_tables_df.append(new_topic_df)
        all_markdown_topic_tables.append(new_markdown_table)

        #display_table = master_summary_response[-1].text

        # Show unique topics alongside document counts as output
        display_table = new_unique_topics_df.to_markdown(index=False)

        whole_conversation_metadata.append(whole_conversation_metadata_str)
        whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)


        # Write final output to text file also
        try:
            new_final_table_output_path = output_folder + master_batch_out_file_part + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

            with open(new_final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                f.write(display_table)

            log_files_output_paths.append(new_final_table_output_path)

        except Exception as e:
            print(e)

    # If this is the first batch, run this
    else:
        #system_prompt_with_table = system_prompt + normalised_simple_markdown_table

        # Prepare Gemini models before query       
        if model_choice in GEMINI_MODELS:
            print("Using Gemini model:", model_choice)
            model, config = construct_gemini_generative_model(in_api_key=in_api_key, temperature=temperature, model_choice=model_choice, system_prompt=system_prompt, max_tokens=max_tokens)
        else:
            print("Using AWS Bedrock model:", model_choice)

        formatted_prompt1 = prompt1.format(response_table=normalised_simple_markdown_table)

        if prompt2: formatted_prompt2 = prompt2.format(response_table=normalised_simple_markdown_table)
        else: formatted_prompt2 = prompt2
        
        if prompt3: formatted_prompt3 = prompt3.format(response_table=normalised_simple_markdown_table)
        else: formatted_prompt3 = prompt3

        batch_prompts = [formatted_prompt1, formatted_prompt2, formatted_prompt3][:number_of_requests]  # Adjust this list to send fewer requests 
        
        #whole_conversation = [system_prompt_with_table]

        whole_conversation = [system_prompt] 

        # Process requests to large language model
        responses, conversation_history, whole_conversation, whole_conversation_metadata = process_requests(batch_prompts, system_prompt, conversation_history, whole_conversation, whole_conversation_metadata, model, config, model_choice, temperature, reported_batch_no)
        
        #print("Whole conversation metadata before:", whole_conversation_metadata)

        topic_table_out_path, reference_table_out_path, unique_topics_df_out_path, topic_table_df, markdown_table, reference_df, new_unique_topics_df, batch_out_file_part, is_error =  write_llm_output_and_logs(responses, whole_conversation, whole_conversation_metadata, out_file_part, latest_batch_completed, start_row, end_row, model_choice_clean, temperature, log_files_output_paths, existing_reference_df, existing_unique_topics_df, first_run=True)

        # If error in table parsing, leave function
        if is_error == True:
            display_table, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_message_out
        
        
        all_topic_tables_df.append(topic_table_df)

        topic_table_df.to_csv(topic_table_out_path, index=None)
        out_file_paths.append(topic_table_out_path)

        reference_df.to_csv(reference_table_out_path, index=None)
        log_files_output_paths.append(reference_table_out_path)

        ## Unique topic list

        new_unique_topics_df = pd.concat([new_unique_topics_df, existing_unique_topics_df]).drop_duplicates('Subtopic')

        new_unique_topics_df.to_csv(unique_topics_df_out_path, index=None)
        out_file_paths.append(unique_topics_df_out_path)
        
        all_markdown_topic_tables.append(markdown_table)

        whole_conversation_metadata.append(whole_conversation_metadata_str)
        whole_conversation_metadata_str = ' '.join(whole_conversation_metadata)
        
        # Write final output to text file also
        try:
            final_table_output_path = output_folder + batch_out_file_part + "_full_final_response_" + model_choice_clean + "_temp_" + str(temperature) + ".txt"

            with open(final_table_output_path, "w", encoding='utf-8', errors='replace') as f:
                f.write(responses[-1].text)

            log_files_output_paths.append(final_table_output_path)

        except Exception as e:
            print(e)

        display_table = responses[-1].text
        new_topic_df = topic_table_df
        new_reference_df = reference_df

    # Increase latest file completed count unless we are at the last file
    if latest_batch_completed != num_batches:
        print("Completed file number:", str(latest_batch_completed))
        latest_batch_completed += 1 

    toc = time.perf_counter()
    final_time = toc - tic
    out_time = f"in {final_time:0.1f} seconds."
    print(out_time)    
    
    out_message.append('All queries successfully completed in')

    final_message_out = '\n'.join(out_message)
    final_message_out = final_message_out + " " + out_time

    final_message_out = final_message_out + "\n\nGo to to the LLM settings tab to see redaction logs. Please give feedback on the results below to help improve this app."    

    return display_table, new_topic_df, new_unique_topics_df, new_reference_df, out_file_paths, out_file_paths, latest_batch_completed, log_files_output_paths, log_files_output_paths, whole_conversation_metadata_str, final_time, final_message_out
