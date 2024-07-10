# TODO manage imports
from btp_proxy_lists import open_ai_models, bedrock_models, gcp_models, huggingface_models

from PyPDF2 import PdfReader

# method to get llm-appropriate embedding
def get_embedding(llm):

    # with BTP proxy issue, only ada works at the moment
    return all_embeddings['text-embedding-ada-002-v2']

    if llm in open_ai_models:
        return all_embeddings['text-embedding-ada-002-v2']
    
    if llm in bedrock_models:
        return all_embeddings['amazon-titan-e1t-medium']
    
    else:
        return all_embeddings['gcp-textembedding-gecko-001']


# takes in pdf file path and returns concatenated text
def pdf_handler(pdf_filepath):

    pdf_reader = PdfReader(pdf_filepath)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text