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

from llm_commons.langchain.proxy.bedrock import BedrockEmbeddings
from llm_commons.langchain.proxy.google import GoogleEmbeddings

import openai
import time

import pandas as pd

import random

from rouge_score import rouge_scorer

max_tokens = 1200
temperature = 0.0
temperature_hf = 0.1

all_llms = {
    'text-davinci-003': OpenAI(deployment_id='text-davinci-003', max_tokens = max_tokens, temperature = temperature),
    'gpt-35-turbo':  ChatOpenAI(deployment_id='gpt-35-turbo', max_tokens = max_tokens, temperature = temperature),
    'gpt-4':  ChatOpenAI(deployment_id='gpt-4', max_tokens = max_tokens, temperature = temperature),
    'gpt-4-32k':  ChatOpenAI(deployment_id='gpt-4-32k', max_tokens = max_tokens, temperature = temperature),
    # 'alephalpha':  AlephAlpha(deployment_id='alephalpha', maximum_tokens=max_tokens, temperature=temperature),
    'anthropic-claude-v1':  BedrockChat(deployment_id='anthropic-claude-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-claude-v2':  BedrockChat(deployment_id='anthropic-claude-v2', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-claude-instant-v1':  BedrockChat(deployment_id='anthropic-claude-instant-v1', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-claude-v1-100k':  BedrockChat(deployment_id='anthropic-claude-v1-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-claude-v2-100k':  BedrockChat(deployment_id='anthropic-claude-v2-100k', model_kwargs={"temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-direct-claude-instant-1':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-instant-1', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'anthropic-direct-claude-2':  BedrockChat(deployment_id='anthropic-direct', model_kwargs={'model': 'claude-2', "temperature": temperature, "max_tokens_to_sample": max_tokens}),
    'ai21-j2-grande-instruct':  Bedrock(deployment_id='ai21-j2-grande-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    'ai21-j2-jumbo-instruct':  Bedrock(deployment_id='ai21-j2-jumbo-instruct', model_kwargs={"temperature": temperature, "maxTokens": max_tokens}),
    'amazon-titan-tg1-large':  Bedrock(deployment_id='amazon-titan-tg1-large', model_kwargs={"temperature": temperature}),
    'gcp-text-bison-001':  GooglePalm(deployment_id='gcp-text-bison-001', temperature = temperature, max_output_tokens = max_tokens),
    'falcon-7b':  HuggingFaceTextGenInference(deployment_id='falcon-7b'),#, temperature = temperature, max_new_tokens = max_tokens), # issue https://github.tools.sap/AI-Playground-Projects/llm-commons/issues/106
    'falcon-40b-instruct':  HuggingFaceTextGenInference(deployment_id='falcon-40b-instruct', temperature = temperature_hf, max_new_tokens = max_tokens),
    'llama2-13b-chat-hf':  HuggingFaceTextGenInference(deployment_id='llama2-13b-chat-hf'),#, temperature = temperature, max_new_tokens = max_tokens),
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
    

def embeddings_with_retry(chunks, embedding, max_retries=10):
    for i in range(max_retries):
        try:
            return FAISS.from_texts(chunks, embedding=embeddings)

        except openai.error.APIError as apie:
            print("ratelimit error, retrying", i, apie)
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)

        except RateLimitError:
            print("ratelimit error, retrying", i, apie)
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)
            
    print("Still hitting rate limit after max retries. Skipping to next")
    
    
def llm_with_retry(docs, query, chain, max_retries=10):
    for i in range(max_retries):
        try:
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                # print(cb)
            print(response)
            return response
        
        except openai.error.APIError as apie:
            print("ratelimit error, retrying", i, apie)
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)
            
        except RuntimeError as re:
            print("threw runtime error, retrying", re)
        except KeyError as ke:
            print("threw key error, retrying", ke)
        except ValueError as ve:
            print("threw value error, retrying", ve)
        except APIError as apie:
            print("threw API error, retrying", apie)
            
    print("Still hitting rate limit after max retries. Skipping to next")


pdf_reader = PdfReader('corpus/pdf/iep/specified/4102400943 plus Templates or Action Context Temp.pdf')
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

query = """
You are the developer of IEP. You are tasked with answering consumer queries on integration. What information is required to start with IEP?
"""

truth = """
To start with IEP, you will need the following information:

Event: The type of event that you want to register or onboard to IEP. This includes the name of the event.

Event Meta: Further details of the event, such as the service name, service type, object details, reported timestamp, and rating.

Action: The list of actions supported by IEP, such as sending an email, starting an operation flow, creating an alert, creating a ticket, or sending a chat notification.

Action Meta: The metadata required for configuring actions, such as the mandatory attributes for each action.

Event Meta: Additional details specific to the use-case, such as the attributes required for the event and their mandatory status, attribute order, and whether value help should be displayed.

Action Context Template: Templates for each action, including placeholders that will be replaced during action execution. For example, an email template may include placeholders for the subject and body.

By providing this information, you can effectively register or onboard a new use-case to IEP.
"""

# Fixed params, varying models

generation_count = 1

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

    # for llm in all_llms:
    for llm in ['falcon-40b-instruct', 'gpt-35-turbo']:
        
        print(llm)
    
        embeddings = get_embedding(llm)
        VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
        docs = VectorStore.similarity_search(query=query, k=5)
        chain = load_qa_chain(llm=all_llms[llm], chain_type="stuff")
        
        result = 'error'
        result = llm_with_retry(docs, query, chain, max_retries=10)
        results[llm] = result
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(truth, result)
        results[llm + '_rouge1_precision'] = scores['rouge1'].precision
        results[llm + '_rouge1_recall'] = scores['rouge1'].recall
        results[llm + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
        results[llm + '_rougeL_precision'] = scores['rougeL'].precision
        results[llm + '_rougeL_recall'] = scores['rougeL'].recall
        results[llm + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure
    
    generations.append(results)

# Fixed model, changing params chunk_size and chunk_overlaps

llm = all_llms['gpt-35-turbo']

chunk_sizes = [200, 400, 600, 800, 1000, 1200, 1400]
chunk_overlaps = [20, 40, 60, 100, 150, 200, 300, 400, 500, 1000]

generations = []

for i in range(generation_count):

    print("going for generation", str(i))

    results = dict()

    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:

            if chunk_size < chunk_overlap:
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            params = "cs_" + str(chunk_size) + "|co_" + str(chunk_overlap)
            print(params)

            
            embeddings = get_embedding(llm)
            VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
            docs = VectorStore.similarity_search(query=query, k=5)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            
            result = 'error'    
            result = llm_with_retry(docs, query, max_retries=10)
            results[params] = result

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            scores = scorer.score(truth, result)

            results[params + '_rouge1_precision'] = scores['rouge1'].precision
            results[params + '_rouge1_recall'] = scores['rouge1'].recall
            results[params + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
            results[params + '_rougeL_precision'] = scores['rougeL'].precision
            results[params + '_rougeL_recall'] = scores['rougeL'].recall
            results[params + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure

    generations.append(results)


# Changing k

llm = all_llms['gpt-35-turbo']

K = list(range(1, 16))

generations = []

for i in range(generation_count):

    print("going for generation", str(i))

    results = dict()

    for k in K:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        params = "k_" + str(k)
        print(params)

        embeddings = get_embedding(llm)
        VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
        docs = VectorStore.similarity_search(query=query, k=k)
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        result = 'error'
        result = llm_with_retry(docs, query, max_retries=10)
        results[params] = result

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(truth, result)

        results[params + '_rouge1_precision'] = scores['rouge1'].precision
        results[params + '_rouge1_recall'] = scores['rouge1'].recall
        results[params + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
        results[params + '_rougeL_precision'] = scores['rougeL'].precision
        results[params + '_rougeL_recall'] = scores['rougeL'].recall
        results[params + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure

    generations.append(results)

# getting top scores

pd.DataFrame(generations).to_csv('results/results_k_rouge.csv')

rouge1_precision_scores = {}
rouge1_recall_scores = {}
rouge1_fmeasure_scores = {}
rougeL_precision_scores = {}
rougeL_recall_scores = {}
rougeL_fmeasure_scores = {}

# all_scoring = [rouge1_precision_scores, rouge1_recall_scores, rouge1_fmeasure_scores, rougeL_precision_scores, rougeL_recall_scores, rougeL_fmeasure_scores]
# all_scoring_str = ['rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure', 'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure']

# turn below snippet into multiple scoring methods, the main scoring method should return the scores
all_scoring = [rouge1_fmeasure_scores, rougeL_fmeasure_scores]
all_scoring_str = ['rouge1_fmeasure', 'rougeL_fmeasure']

for generation in generations:
    for column in generation:
        for i in range(len(all_scoring)):
            if all_scoring_str[i] in column:
                score = generation[column]
                if score in all_scoring[i]:
                    all_scoring[i][score].append(column)
                else:
                    all_scoring[i][score] = [column]

for scoring in all_scoring:
    sorted_scores = sorted(scoring)[::-1]
    for i in range(1, 10):
        print(scoring[sorted_scores[i]], sorted_scores[i])
    print()