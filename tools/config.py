import logging
import os
import socket
import tempfile
from datetime import datetime

from dotenv import load_dotenv

today_rev = datetime.now().strftime("%Y%m%d")
HOST_NAME = socket.gethostname()

# Set or retrieve configuration variables for the redaction app


def get_or_create_env_var(var_name: str, default_value: str, print_val: bool = False):
    """
    Get an environmental variable, and set it to a default value if it doesn't exist
    """
    # Get the environment variable if it exists
    value = os.environ.get(var_name)

    # If it doesn't exist, set the environment variable to the default value
    if value is None:
        os.environ[var_name] = default_value
        value = default_value

    if print_val:
        print(f"The value of {var_name} is {value}")

    return value


def ensure_folder_exists(output_folder: str):
    """Checks if the specified folder exists, creates it if not."""

    if not os.path.exists(output_folder):
        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created the {output_folder} folder.")
    else:
        print(f"The {output_folder} folder already exists.")


def add_folder_to_path(folder_path: str):
    """
    Check if a folder exists on your system. If so, get the absolute path and then add it to the system Path variable if it doesn't already exist. Function is only relevant for locally-created executable files based on this app (when using pyinstaller it creates a _internal folder that contains tesseract and poppler. These need to be added to the system path to enable the app to run)
    """

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(folder_path, "folder exists.")

        # Resolve relative path to absolute path
        absolute_path = os.path.abspath(folder_path)

        current_path = os.environ["PATH"]
        if absolute_path not in current_path.split(os.pathsep):
            full_path_extension = absolute_path + os.pathsep + current_path
            os.environ["PATH"] = full_path_extension
            # print(f"Updated PATH with: ", full_path_extension)
        else:
            print(f"Directory {folder_path} already exists in PATH.")
    else:
        print(f"Folder not found at {folder_path} - not added to PATH")


def convert_string_to_boolean(value: str) -> bool:
    """Convert string to boolean, handling various formats."""
    if isinstance(value, bool):
        return value
    elif value in ["True", "1", "true", "TRUE"]:
        return True
    elif value in ["False", "0", "false", "FALSE"]:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {value}")


ensure_folder_exists("config/")

# If you have an aws_config env file in the config folder, you can load in app variables this way, e.g. 'config/app_config.env'
APP_CONFIG_PATH = get_or_create_env_var(
    "APP_CONFIG_PATH", "config/app_config.env"
)  # e.g. config/app_config.env

if APP_CONFIG_PATH:
    if os.path.exists(APP_CONFIG_PATH):
        print(f"Loading app variables from config file {APP_CONFIG_PATH}")
        load_dotenv(APP_CONFIG_PATH)
    else:
        print("App config file not found at location:", APP_CONFIG_PATH)

# Report logging to console?
LOGGING = get_or_create_env_var("LOGGING", "False")

if LOGGING == "True":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

###
# AWS CONFIG
###

# If you have an aws_config env file in the config folder, you can load in AWS keys this way, e.g. 'env/aws_config.env'
AWS_CONFIG_PATH = get_or_create_env_var(
    "AWS_CONFIG_PATH", ""
)  # e.g. config/aws_config.env

if AWS_CONFIG_PATH:
    if os.path.exists(AWS_CONFIG_PATH):
        print(f"Loading AWS variables from config file {AWS_CONFIG_PATH}")
        load_dotenv(AWS_CONFIG_PATH)
    else:
        print("AWS config file not found at location:", AWS_CONFIG_PATH)

RUN_AWS_FUNCTIONS = get_or_create_env_var("RUN_AWS_FUNCTIONS", "0")

AWS_REGION = get_or_create_env_var("AWS_REGION", "")

AWS_DEFAULT_REGION = get_or_create_env_var("AWS_DEFAULT_REGION", "")

AWS_CLIENT_ID = get_or_create_env_var("AWS_CLIENT_ID", "")

AWS_CLIENT_SECRET = get_or_create_env_var("AWS_CLIENT_SECRET", "")

AWS_USER_POOL_ID = get_or_create_env_var("AWS_USER_POOL_ID", "")

AWS_ACCESS_KEY = get_or_create_env_var("AWS_ACCESS_KEY", "")
if AWS_ACCESS_KEY:
    print("AWS_ACCESS_KEY found in environment variables")

AWS_SECRET_KEY = get_or_create_env_var("AWS_SECRET_KEY", "")
if AWS_SECRET_KEY:
    print("AWS_SECRET_KEY found in environment variables")

# Bedrock Knowledge Base retrieve_and_generate (bypasses local FAISS RAG when enabled).
# Requires IAM permissions for bedrock:RetrieveAndGenerate / knowledge-base retrieve,
# plus access to the foundation model. Credentials follow the same pattern as RUN_AWS_FUNCTIONS.
USE_BEDROCK_KB = get_or_create_env_var("USE_BEDROCK_KB", "0")
KNOWLEDGE_BASE_ID = get_or_create_env_var("KNOWLEDGE_BASE_ID", "")
BEDROCK_MODEL_ID = get_or_create_env_var("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
GUARDRAIL_ID = get_or_create_env_var("GUARDRAIL_ID", "")
GUARDRAIL_VERSION = get_or_create_env_var("GUARDRAIL_VERSION", "")
if USE_BEDROCK_KB == "1":
    print(
        "USE_BEDROCK_KB=1: local FAISS retrieval/generation bypassed; "
        f"KB_ID={KNOWLEDGE_BASE_ID or '(not set)'}, model={BEDROCK_MODEL_ID}"
    )

# Arize Phoenix / AX tracing for chat retrieve+generate turns (custom OTEL spans).
# Local Phoenix: ARIZE_TRACING_ENABLED=1, ARIZE_BACKEND=phoenix,
# PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 (default).
ARIZE_TRACING_ENABLED = get_or_create_env_var("ARIZE_TRACING_ENABLED", "0")
ARIZE_BACKEND = get_or_create_env_var("ARIZE_BACKEND", "phoenix")
PHOENIX_COLLECTOR_ENDPOINT = get_or_create_env_var(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
)
PHOENIX_PROJECT_NAME = get_or_create_env_var(
    "PHOENIX_PROJECT_NAME", "light-pdf-qa-chatbot"
)
ARIZE_PROJECT_NAME = get_or_create_env_var("ARIZE_PROJECT_NAME", "")
PHOENIX_API_KEY = get_or_create_env_var("PHOENIX_API_KEY", "")
ARIZE_SPACE_ID = get_or_create_env_var("ARIZE_SPACE_ID", "")
ARIZE_API_KEY = get_or_create_env_var("ARIZE_API_KEY", "")
ARIZE_ENDPOINT = get_or_create_env_var("ARIZE_ENDPOINT", "europe")
if ARIZE_TRACING_ENABLED in {"1", "true", "True", "yes", "on"}:
    print(
        f"ARIZE_TRACING_ENABLED: backend={ARIZE_BACKEND}, "
        f"phoenix_endpoint={PHOENIX_COLLECTOR_ENDPOINT}, "
        f"project={PHOENIX_PROJECT_NAME or ARIZE_PROJECT_NAME or 'light-pdf-qa-chatbot'}"
    )

QA_CHATBOT_BUCKET = get_or_create_env_var("QA_CHATBOT_BUCKET", "")

# Upload access/feedback logs to S3 (requires RUN_AWS_FUNCTIONS=1 and QA_CHATBOT_BUCKET)
SAVE_LOGS_TO_S3 = get_or_create_env_var("SAVE_LOGS_TO_S3", "False")

# Custom headers e.g. if routing traffic through Cloudfront
# Retrieving or setting CUSTOM_HEADER
CUSTOM_HEADER = get_or_create_env_var("CUSTOM_HEADER", "")
# if CUSTOM_HEADER: print(f'CUSTOM_HEADER found')

# Retrieving or setting CUSTOM_HEADER_VALUE
CUSTOM_HEADER_VALUE = get_or_create_env_var("CUSTOM_HEADER_VALUE", "")
# if CUSTOM_HEADER_VALUE: print(f'CUSTOM_HEADER_VALUE found')

###
# File I/O config
###
SESSION_OUTPUT_FOLDER = get_or_create_env_var(
    "SESSION_OUTPUT_FOLDER", "False"
)  # i.e. do you want your input and output folders saved within a subfolder based on session hash value within output/input folders

OUTPUT_FOLDER = get_or_create_env_var("GRADIO_OUTPUT_FOLDER", "output/")  # 'output/'
INPUT_FOLDER = get_or_create_env_var("GRADIO_INPUT_FOLDER", "input/")  # 'input/'

ensure_folder_exists(OUTPUT_FOLDER)
ensure_folder_exists(INPUT_FOLDER)

# Allow for files to be saved in a temporary folder for increased security in some instances
if OUTPUT_FOLDER == "TEMP" or INPUT_FOLDER == "TEMP":
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")

        if OUTPUT_FOLDER == "TEMP":
            OUTPUT_FOLDER = temp_dir + "/"
        if INPUT_FOLDER == "TEMP":
            INPUT_FOLDER = temp_dir + "/"

# By default, logs are put into a subfolder of today's date and the host name of the instance running the app. This is to avoid at all possible the possibility of log files from one instance overwriting the logs of another instance on S3. If running the app on one system always, or just locally, it is not necessary to make the log folders so specific.
# Another way to address this issue would be to write logs to another type of storage, e.g. database such as dynamodb. I may look into this in future.

USE_LOG_SUBFOLDERS = get_or_create_env_var("USE_LOG_SUBFOLDERS", "True")

if USE_LOG_SUBFOLDERS == "True":
    day_log_subfolder = today_rev + "/"
    host_name_subfolder = HOST_NAME + "/"
    full_log_subfolder = day_log_subfolder + host_name_subfolder
else:
    full_log_subfolder = ""

FEEDBACK_LOGS_FOLDER = get_or_create_env_var(
    "FEEDBACK_LOGS_FOLDER", "feedback/" + full_log_subfolder
)
ACCESS_LOGS_FOLDER = get_or_create_env_var(
    "ACCESS_LOGS_FOLDER", "logs/" + full_log_subfolder
)
USAGE_LOGS_FOLDER = get_or_create_env_var(
    "USAGE_LOGS_FOLDER", "usage/" + full_log_subfolder
)

ensure_folder_exists(FEEDBACK_LOGS_FOLDER)
ensure_folder_exists(ACCESS_LOGS_FOLDER)
ensure_folder_exists(USAGE_LOGS_FOLDER)

# Should the redacted file name be included in the logs? In some instances, the names of the files themselves could be sensitive, and should not be disclosed beyond the app. So, by default this is false.
DISPLAY_FILE_NAMES_IN_LOGS = get_or_create_env_var(
    "DISPLAY_FILE_NAMES_IN_LOGS", "False"
)

###
# RUN CONFIG
RUN_GEMINI_MODELS = get_or_create_env_var("RUN_GEMINI_MODELS", "1")

GEMINI_API_KEY = get_or_create_env_var("GEMINI_API_KEY", "")

# NOTE THAT THIS IS REQUIRED

HF_TOKEN = get_or_create_env_var("HF_TOKEN", "")


# Number of pages to loop through before breaking the function and restarting from the last finished page (not currently activated).
PAGE_BREAK_VALUE = get_or_create_env_var("PAGE_BREAK_VALUE", "99999")

MAX_TIME_VALUE = get_or_create_env_var("MAX_TIME_VALUE", "999999")

###
# APP RUN CONFIG
###

FILL_SCREEN_WIDTH = convert_string_to_boolean(
    get_or_create_env_var("FILL_SCREEN_WIDTH", "False")
)

SMALL_MODEL_NAME = get_or_create_env_var("SMALL_MODEL_NAME", "Qwen 3.5 0.8B")

SMALL_MODEL_REPO_ID = get_or_create_env_var(
    "SMALL_MODEL_REPO_ID", "unsloth/Qwen3.5-0.8B"
)

# Local Hugging Face generate: abandon if no first token within this many seconds.
GENERATION_FIRST_TOKEN_TIMEOUT = float(
    get_or_create_env_var("GENERATION_FIRST_TOKEN_TIMEOUT", "45")
)
# Subsequent tokens: max wait between streamed chunks before aborting.
GENERATION_TOKEN_TIMEOUT = float(
    get_or_create_env_var("GENERATION_TOKEN_TIMEOUT", "120")
)

LOAD_LARGE_MODEL = get_or_create_env_var("LOAD_LARGE_MODEL", "0")

LARGE_MODEL_NAME = get_or_create_env_var(
    "LARGE_MODEL_NAME", "Phi 3.5 Mini (larger, slow)"
)

LARGE_MODEL_REPO_ID = get_or_create_env_var(
    "LARGE_MODEL_REPO_ID", "QuantFactory/Phi-3.5-mini-instruct-GGUF"
)  # THIS METHOD IS DEPRECATED AND WILL NO LONGER BE USED IN FUTURE (Llama-cpp-python is no longer being updated)

LARGE_MODEL_GGUF_FILE = get_or_create_env_var(
    "LARGE_MODEL_GGUF_FILE", "Phi-3.5-mini-instruct.Q4_K_M.gguf"
)

# Build up options for models, grouped by provider for the UI
LOCAL_MODELS = [SMALL_MODEL_NAME]
if LOAD_LARGE_MODEL == "1":
    LOCAL_MODELS.append(LARGE_MODEL_NAME)

# Always defined so callers can safely do membership checks
AWS_MODELS = []
if RUN_AWS_FUNCTIONS == "1":
    AWS_MODELS = [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-sonnet-4-6",
        "amazon.nova-micro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-pro-v1:0",
        "deepseek.v3-v1:0",
        "openai.GPT-OSS 20B-1:0",
        "openai.gpt-oss-120b-1:0",
        "google.gemma-3-12b-it",
        "google.gemma-3-27b-it",
        "mistral.ministral-3-14b-instruct",
        "mistral.devstral-2-123b",
        "nvidia.nemotron-super-3-120b",
        "mistral.magistral-small-2509",
    ]

GEMINI_MODELS = []
if RUN_GEMINI_MODELS == "1":
    GEMINI_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-pro-latest",
    ]

# Provider label -> model list (only include providers that have models available)
PROVIDER_MODELS = {"Local": LOCAL_MODELS}
if GEMINI_MODELS:
    PROVIDER_MODELS["Google"] = GEMINI_MODELS
if AWS_MODELS:
    PROVIDER_MODELS["AWS"] = AWS_MODELS

MODEL_PROVIDER_CHOICES = list(PROVIDER_MODELS.keys())
DEFAULT_MODEL_PROVIDER = "Local"

default_model_choices = LOCAL_MODELS + AWS_MODELS + GEMINI_MODELS

DEFAULT_MODEL_CHOICES = get_or_create_env_var(
    "DEFAULT_MODEL_CHOICES", str(default_model_choices)
)

EMBEDDINGS_MODEL_NAME = get_or_create_env_var(
    "EMBEDDINGS_MODEL_NAME", "mixedbread-ai/mxbai-embed-xsmall-v1"
)  # "mixedbread-ai/mxbai-embed-xsmall-v1"

DEFAULT_EMBEDDINGS_LOCATION = get_or_create_env_var(
    "DEFAULT_EMBEDDINGS_LOCATION", "faiss_embedding/faiss_embedding.zip"
)

DEFAULT_DATA_SOURCE_NAME = get_or_create_env_var(
    "DEFAULT_DATA_SOURCE_NAME", "Document redaction app documentation"
)

DEFAULT_DATA_SOURCE = get_or_create_env_var(
    "DEFAULT_DATA_SOURCE",
    "https://seanpedrick-case.github.io/doc_redaction/src/user_guide.html",
)

DEFAULT_EXAMPLES = get_or_create_env_var(
    "DEFAULT_EXAMPLES",
    '[ "How can I make a custom deny list?", "How can I find duplicate pages in a document?", "How can I review and modify existing redactions?", "How can I export my review files to Adobe?"]',
)
#
# ') # ["What were the five pillars of the previous borough plan?",
# "What is the vision statement for Lambeth?",
# "What are the commitments for Lambeth?",
# "What are the 2030 outcomes for Lambeth?"]

# Get some environment variables and Launch the Gradio app
COGNITO_AUTH = get_or_create_env_var("COGNITO_AUTH", "0")

RUN_DIRECT_MODE = get_or_create_env_var("RUN_DIRECT_MODE", "0")

MAX_QUEUE_SIZE = int(get_or_create_env_var("MAX_QUEUE_SIZE", "5"))

MAX_FILE_SIZE = get_or_create_env_var("MAX_FILE_SIZE", "250mb")

GRADIO_SERVER_PORT = int(get_or_create_env_var("GRADIO_SERVER_PORT", "7860"))

ROOT_PATH = get_or_create_env_var("ROOT_PATH", "")

DEFAULT_CONCURRENCY_LIMIT = get_or_create_env_var("DEFAULT_CONCURRENCY_LIMIT", "3")

GET_DEFAULT_ALLOW_LIST = get_or_create_env_var("GET_DEFAULT_ALLOW_LIST", "False")

ALLOW_LIST_PATH = get_or_create_env_var(
    "ALLOW_LIST_PATH", ""
)  # config/default_allow_list.csv

S3_ALLOW_LIST_PATH = get_or_create_env_var(
    "S3_ALLOW_LIST_PATH", ""
)  # default_allow_list.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

if ALLOW_LIST_PATH:
    OUTPUT_ALLOW_LIST_PATH = ALLOW_LIST_PATH
else:
    OUTPUT_ALLOW_LIST_PATH = "config/default_allow_list.csv"

SHOW_COSTS = get_or_create_env_var("SHOW_COSTS", "False")

GET_COST_CODES = get_or_create_env_var("GET_COST_CODES", "False")

DEFAULT_COST_CODE = get_or_create_env_var("DEFAULT_COST_CODE", "")

COST_CODES_PATH = get_or_create_env_var(
    "COST_CODES_PATH", ""
)  # 'config/COST_CENTRES.csv' # file should be a csv file with a single table in it that has two columns with a header. First column should contain cost codes, second column should contain a name or description for the cost code

S3_COST_CODES_PATH = get_or_create_env_var(
    "S3_COST_CODES_PATH", ""
)  # COST_CENTRES.csv # This is a path within the DOCUMENT_REDACTION_BUCKET

if COST_CODES_PATH:
    OUTPUT_COST_CODES_PATH = COST_CODES_PATH
else:
    OUTPUT_COST_CODES_PATH = "config/COST_CENTRES.csv"

ENFORCE_COST_CODES = get_or_create_env_var(
    "ENFORCE_COST_CODES", "False"
)  # If you have cost codes listed, is it compulsory to choose one before redacting?

if ENFORCE_COST_CODES == "True":
    GET_COST_CODES = "True"
