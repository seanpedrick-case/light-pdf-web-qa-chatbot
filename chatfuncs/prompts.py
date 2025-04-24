instruction_prompt_template_alpaca_quote = """### Instruction:
Quote directly from the SOURCE below that best answers the QUESTION. Only quote full sentences in the correct order. If you cannot find an answer, start your response with "My best guess is: ".
    
CONTENT: {summaries}   
QUESTION: {question}

Response:"""

instruction_prompt_template_alpaca = """### Instruction:
### User:
Answer the QUESTION using information from the following CONTENT.
CONTENT: {summaries}
QUESTION: {question}

Response:"""


instruction_prompt_template_wizard_orca = """### HUMAN:
Answer the QUESTION below based on the CONTENT. Only refer to CONTENT that directly answers the question.
CONTENT - {summaries}
QUESTION - {question}
### RESPONSE:
"""


instruction_prompt_template_orca = """### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
Answer the QUESTION with a short response using information from the following CONTENT.
QUESTION: {question}
CONTENT: {summaries}

### Response:"""

instruction_prompt_template_orca_quote = """### System:
You are an AI assistant that follows instruction extremely well. Help as much as you can.
### User:
Quote text from the CONTENT to answer the QUESTION below.
QUESTION: {question}
CONTENT: {summaries}  
### Response:
"""


instruction_prompt_mistral_orca = """<|im_start|>system\n
You are an AI assistant that follows instruction extremely well. Help as much as you can.
<|im_start|>user\n
Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.
CONTENT: {summaries}
QUESTION: {question}\n
Answer:<|im_end|>"""

instruction_prompt_phi3 = """<|user|>\n
Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.\n
CONTENT: {summaries}\n
QUESTION: {question}\n
Answer:<|end|>\n
<|assistant|>"""

instruction_prompt_llama3 = """<|start_header_id|>system<|end_header_id|>\n
You are an AI assistant that follows instruction extremely well. Help as much as you can.<|eot_id|><|start_header_id|>user<|end_header_id|>\n
Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.\n
CONTENT: {summaries}\n
QUESTION: {question}\n
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""

instruction_prompt_qwen = """<|im_start|>system\n
You are an AI assistant that follows instruction extremely well. Help as much as you can.
<|im_start|>user\n
Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.
CONTENT: {summaries}
QUESTION: {question}\n
Answer:<|im_end|>
<|im_start|>assistant\n"""

instruction_prompt_gemma = """Answer the QUESTION using information from the following CONTENT. Respond with short answers that directly answer the question.
CONTENT: {summaries}
QUESTION: {question}
assistant:"""