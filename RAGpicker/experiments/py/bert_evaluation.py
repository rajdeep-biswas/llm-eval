import torch
import pandas as pd
from bert_score import score
from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from llm_commons.langchain.proxy.openai import ChatOpenAI
from llm_commons.langchain.proxy.bedrock import BedrockChat, Bedrock
from llm_commons.langchain.proxy.google import GooglePalm
from llm_commons.langchain.proxy.huggingface import HuggingFaceTextGenInference
from llm_commons.langchain.proxy.aleph_alpha import AlephAlpha

from llm_commons.langchain.proxy.openai import OpenAI
from llm_commons.langchain.proxy import OpenAIEmbeddings

import numpy as np 
import matplotlib.pyplot as plt 

from llm_commons.langchain.proxy.bedrock import BedrockEmbeddings
from llm_commons.langchain.proxy.google import GoogleEmbeddings
from rouge_score import rouge_scorer
from llm_commons.proxy.base import set_proxy_version
from llm_commons.btp_llm.identity import BTPProxyClient
from llm_commons.langchain.proxy import init_llm, init_embedding_model
set_proxy_version('btp')
 
btp_proxy_client = BTPProxyClient()
#  BERT independent implementation starts---------------------------------------------------
# # import pandas lib as pd
# import pandas as pd
 
# # read by default 1st sheet of an excel file
# df = pd.read_excel('/Users/I587795/Downloads/iep_CS20230006563076_all_models_rouge_withBERT.xls')
 
# print(df)

# ref_text = df['Ground Truth'].iloc(0)

# all_llms = ['text-davinci-003','gpt-35-turbo','gpt-4','gpt-4-32k','alephalpha','anthropic-claude-v1','anthropic-claude-v2','anthropic-claude-instant-v1','anthropic-claude-v1-100k','anthropic-claude-v2-100k','anthropic-direct-claude-instant-1','anthropic-direct-claude-2','ai21-j2-grande-instruct','ai21-j2-jumbo-instruct','amazon-titan-tg1-large','gcp-text-bison-001','falcon-7b','falcon-40b-instruct','llama2-13b-chat-hf']

# for i in range(0, len(all_llms)):
#     df.loc[:,all_llms[i]+'_BERT'] =  0

# print(df.columns)
# for row in range(0,3):
#     for i in range(0, len(all_llms)):
#         # compute Bert score
#         P, R, F1 = score(df[all_llms[i]], ref_text, lang="en", model_type="microsoft/deberta-xlarge-mnli")

#     # print results
#         print(f"Bert score: P={P.item():.4f} R={R.item():.4f} F1={F1.item():.4f}")
        



# # reference and generated texts
# ref_text = """
# The root cause of the problem appears to be related to IP address allowlisting and the configuration of the Operations URI.
# It seems that the Operations URI may need to be left empty according to the documentation provided.
# Additionally, there is uncertainty about the correct format for the Operations URI and whether the IP address for SAP DCs has been allowlisted in the MDI service server.
# These issues may be contributing to the connection error. It is recommended to confirm the correct configuration for the Operations URI and ensure that the necessary IP address allowlisting has been completed.
# """
# gen_text = """
# The error message indicates that the destination server is not able to find the specified URL. This could be due to a few reasons:

# 1. The URL is incorrect or outdated.
# 2. The destination server is not configured to allow access to the URL.
# 3. The destination server is down or experiencing issues.

# To resolve the issue, you can try the following steps:

# 1. Double-check the URL mentioned in the error message and ensure that it is correct and up-to-date.
# 2. Check with your IT team to ensure that the destination server is configured to allow access to the URL.
# 3. If the destination server is down or experiencing issues, wait until it is back up and running before attempting to connect again.

# If none of these steps work, you may need to contact SAP support for further assistance.
# """

# # compute Bert score
# P, R, F1 = score([gen_text], [ref_text], lang="en", model_type="microsoft/deberta-xlarge-mnli")

# # print results
# print(f"Bert score: P={P.item():.4f} R={R.item():.4f} F1={F1.item():.4f}") ------------------------------------------ends





#Model parameters 
max_tokens = 800
temperature = 0
temperature_hf = 0.1

all_llms = {
    'text-davinci-003': OpenAI(deployment_id='text-davinci-003', max_tokens = max_tokens, temperature = temperature),
    'gpt-35-turbo':  ChatOpenAI(deployment_id='gpt-35-turbo', max_tokens = max_tokens, temperature = temperature),
    'gpt-4':  ChatOpenAI(deployment_id='gpt-4', max_tokens = max_tokens, temperature = temperature),
    'gpt-4-32k':  ChatOpenAI(deployment_id='gpt-4-32k', max_tokens = max_tokens, temperature = temperature),
    # 'alephalpha':  AlephAlpha(deployment_id='alephalpha', maximum_tokens=max_tokens, temperature=temperature),
    # 'anthropic-claude-v1':  BedrockChat(deployment_id='anthropic-claude-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v2':  BedrockChat(deployment_id='anthropic-claude-v2', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-instant-v1':  BedrockChat(deployment_id='anthropic-claude-instant-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v1-100k':  BedrockChat(deployment_id='anthropic-claude-v1-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-claude-v2-100k':  BedrockChat(deployment_id='anthropic-claude-v2-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-direct-claude-instant-1':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-instant-1', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'anthropic-direct-claude-2':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-2', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    # 'ai21-j2-grande-instruct':  Bedrock(deployment_id='ai21-j2-grande-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    # 'ai21-j2-jumbo-instruct':  Bedrock(deployment_id='ai21-j2-jumbo-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    # 'amazon-titan-tg1-large':  Bedrock(deployment_id='amazon-titan-tg1-large', model_kwargs={"temperature": temperature}),
    # 'gcp-text-bison-001':  GooglePalm(deployment_id='gcp-text-bison-001', temperature = temperature, max_output_tokens = max_tokens),
    # 'falcon-7b':  HuggingFaceTextGenInference(deployment_id='falcon-7b'),#, temperature = temperature, max_new_tokens = max_tokens), # issue https://github.tools.sap/AI-Playground-Projects/llm-commons/issues/106
    'falcon-40b-instruct':  HuggingFaceTextGenInference(deployment_id='falcon-40b-instruct', temperature = temperature_hf, max_new_tokens = max_tokens),
    # 'llama2-13b-chat-hf':  HuggingFaceTextGenInference(deployment_id='llama2-13b-chat-hf'),#, temperature = temperature, max_new_tokens = max_tokens),
}

all_embeddings = {
    'text-embedding-ada-002-v2':  OpenAIEmbeddings(deployment_id='text-embedding-ada-002-v2'),
    'amazon-titan-e1t-medium':  BedrockEmbeddings(deployment_id='amazon-titan-e1t-medium'),
    'gcp-textembedding-gecko-001':  GoogleEmbeddings(deployment_id='gcp-textembedding-gecko-001'),
}

open_ai_models = set([
    'text-davinci-003',
    'gpt-35-turbo',
    'gpt-4',
    'gpt-4-32k',
    'alephalpha'
])

bedrock_models = set([
    'anthropic-claude-v1',
    'anthropic-claude-v2',
    'anthropic-claude-instant-v1',
    'anthropic-claude-v1-100k',
    'anthropic-claude-v2-100k',
    'anthropic-direct-claude-instant-1',
    'anthropic-direct-claude-2',
    'ai21-j2-grande-instruct',
    'ai21-j2-jumbo-instruct',
    'amazon-titan-tg1-large'
])

gcp_models = set([
    'gcp-text-bison-001'
])

huggingfacemodels = set([
    'falcon-7b',
    'falcon-40b-instruct',
    'llama2-13b-chat-hf'
])

def get_embedding(llm):
    return all_embeddings['text-embedding-ada-002-v2']
    if llm in open_ai_models:
        return all_embeddings['text-embedding-ada-002-v2']
    if llm in bedrock_models:
        return all_embeddings['amazon-titan-e1t-medium']
    else:
        return all_embeddings['gcp-textembedding-gecko-001']

pdf_reader = PdfReader('/Users/I587795/Desktop/GenAI@CALM/SF_EC_ONEmds_Int_en-US.pdf')
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

query = """
How should I proceed to resolve error:
 
Error: Could not connect to destination server. Please check to see if IP Address mentioned in https://launchpad.support.sap.com/#/notes/2395508 is allowlisted by your destination server. Contact your IT for more information
 
HTTP Error Response code 404 with response state text: Not found while calling Operations url: 'https//.../sap.odm.finance.costobject.costcenter. No response options available.
"""

truth = """
The root cause of the problem appears to be related to IP address allowlisting and the configuration of the Operations URI.
It seems that the Operations URI may need to be left empty according to the documentation provided.
Additionally, there is uncertainty about the correct format for the Operations URI and whether the IP address for SAP DCs has been allowlisted in the MDI service server.
These issues may be contributing to the connection error. It is recommended to confirm the correct configuration for the Operations URI and ensure that the necessary IP address allowlisting has been completed.
"""

generation_count = 3

generations = []

for i in range(generation_count):
    
    print("going for generation", str(i))
    
    results = dict()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    for llm in all_llms:
    
        embeddings = get_embedding(llm)
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        docs = VectorStore.similarity_search(query=query, k=5)
        chain = load_qa_chain(llm=all_llms[llm], chain_type="stuff")
        
        result = 'error'
        print(llm)
        try: 
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                # print(cb)
            print(response)
            result = response
        except RuntimeError as re:
            print("threw error", re)
        except KeyError as ke:
            print("threw error", ke)
        except ValueError as ve:
            print("threw error", ve)
        results[llm] = result
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(truth, result)
        P, R, F1 = score([result], [truth], lang="en", model_type="microsoft/deberta-xlarge-mnli")
        print(f"Bert score: P={P.item():.4f} R={R.item():.4f} F1={F1.item():.4f}")
        results[llm + '_rouge1_precision'] = scores['rouge1'].precision
        results[llm + '_rouge1_recall'] = scores['rouge1'].recall
        results[llm + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
        results[llm + '_rougeL_precision'] = scores['rougeL'].precision
        results[llm + '_rougeL_recall'] = scores['rougeL'].recall
        results[llm + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure
        results[llm + '_BERT_precision'] = P.item()
        results[llm + '_BERT_recall'] = R.item()
        results[llm + '_BERT_fmeasure'] = F1.item()

    generations.append(results)
    length = len(results[llm + '_rouge1_precision'])
    barwidth = 0.25
    fig = plt.subplots(figsize=(12,8))
    br1 = np.arrange(length)
    br2 = [x+ barwidth for x in br1]
    br3 = [x+ barwidth for x in br2]
    plt.bar(br1, results[llm + '_rouge1_fmeasure'], color ='r', width = barwidth, edgecolor ='grey', label ='Rouge1_F1score') 
    plt.bar(br2, results[llm + '_rougeL_fmeasure'], color ='g', width = barwidth, edgecolor ='grey', label ='RougeL_F1score') 
    plt.bar(br3, results[llm + '_BERT_fmeasure'], color ='b', width = barwidth, edgecolor ='grey', label ='BERT_Fmeasure') 
 
    # Adding Xticks 
    plt.xlabel('Models', fontweight ='bold', fontsize = 15) 
    plt.ylabel('F1Score', fontweight ='bold', fontsize = 15) 
    plt.xticks([r + barwidth for r in range(len(results[llm + '_rougeL_fmeasure']))], all_llms.keys)
    
    plt.legend()
    plt.show() 
    plt.savefig("output_"+generation_count+"_.jpg")

df = pd.DataFrame(generations)
df.insert(0, "Prompt", query)
df.insert(1, "Ground Truth", truth)
df.to_excel('sf_all_models_rouge_BERT.xlsx')
