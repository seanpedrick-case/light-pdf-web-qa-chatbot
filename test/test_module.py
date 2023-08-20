# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pytest
import gradio as gr
from ..chatfuncs.ingest import *
from ..chatfuncs.chatfuncs import *

def test_read_docx():
    content = read_docx('sample.docx')
    assert content == "Hello, World!"


# +
def test_parse_file():
    # Assuming these files exist and you know their content
    files = ['sample.docx', 'sample.pdf', 'sample.txt', 'sample.html']
    contents = parse_file(files)
    
    assert contents['sample.docx'] == 'Hello, World!'
    assert contents['sample.pdf'] == 'Hello, World!'
    assert contents['sample.txt'] == 'Hello, World!'
    assert contents['sample.html'] == 'Hello, World!'

def test_unsupported_file_type():
    files = ['sample.unknown']
    contents = parse_file(files)
    assert contents['sample.unknown'].startswith('Unsupported file type:')

def test_input_validation():
    with pytest.raises(ValueError, match="Expected a list of file paths."):
        parse_file('single_file_path.txt')