# TODO create btp_proxy_lists.py file which holds dictionaries of all llms and embeddings
from btp_proxy_lists import all_llms, all_embeddings

from helper import get_embedding, pdf_handler

# TODO organize imports
"""
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

from rouge_score import rouge_scorer
"""

class RAGInput:

    corpus_text = None
    ground_truth = None

    # TODO figure out how to pass these through the constructor

    def set_corpus(self, pdf_filepath):
        self.corpus_text = pdf_handler(pdf_filepath)

    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth


class RAGPicker:

    max_tokens = None
    temperature = None

    # huggingface models only allows temperature value > 0.0
    temperature_hf = None

    # constructor method to initialize parameters
    def __init__(self, max_tokens = 800, temperature = 0):
        # TODO find better way to initiate parameters
        self.max_tokens = max_tokens
        self.temperature = temperature

        # set temperature_hf to 0.1 if temperature is 0, else, set it to temperature
        self.temperature_hf = max(0.1, temperature)
    
    # TODO prompt and generation handling
        
    # method to evaluate all models
    
    def evaluate_all_models(self, prompt, generation_count = 3, k = 5, chunk_size = 1000, chunk_overlap = 200):
        generations = []

        for i in range(generation_count):
            
            print("going for generation", str(i))
            
            results = dict()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            for llm in all_llms:
            
                embeddings = get_embedding(llm)
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                docs = VectorStore.similarity_search(prompt=prompt, k=k)
                chain = load_qa_chain(llm=all_llms[llm], chain_type="stuff")
                
                result = 'error'
                print(llm)
                try: 
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=prompt)
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
                results[llm + '_rouge1_precision'] = scores['rouge1'].precision
                results[llm + '_rouge1_recall'] = scores['rouge1'].recall
                results[llm + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
                results[llm + '_rougeL_precision'] = scores['rougeL'].precision
                results[llm + '_rougeL_recall'] = scores['rougeL'].recall
                results[llm + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure
            
            generations.append(results)