import random
import time

import openai

from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback

from model_manager import ModelManager

model_manager = ModelManager(max_tokens = 1200, temperature = 0.0, temperature_hf = 0.1)

def get_embedding(llm):

    all_embeddings = model_manager.all_embeddings

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
            return FAISS.from_texts(chunks, embedding=embedding)

        except openai.error.APIError as apie:
            print("ratelimit error, retrying", i, apie)
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)

        """
        except RateLimitError:
            print("ratelimit error, retrying", i, apie)
            wait_time = (2 ** i) + random.random()
            time.sleep(wait_time)
        """
            
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
        # except APIError as apie:
        #     print("threw API error, retrying", apie)
            
    print("Still hitting rate limit after max retries. Skipping to next")