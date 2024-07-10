from llm_commons.langchain.proxy.openai import ChatOpenAI
from llm_commons.langchain.proxy.bedrock import BedrockChat, Bedrock
from llm_commons.langchain.proxy.google import GooglePalm
from llm_commons.langchain.proxy.huggingface import HuggingFaceTextGenInference
from llm_commons.langchain.proxy.aleph_alpha import AlephAlpha

from llm_commons.langchain.proxy.openai import OpenAI
from llm_commons.langchain.proxy import OpenAIEmbeddings

from llm_commons.langchain.proxy.bedrock import BedrockEmbeddings
from llm_commons.langchain.proxy.google import GoogleEmbeddings

# import openai

from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader

class ModelManager:
    def __init__(self, max_tokens, temperature, temperature_hf):
        self.all_llms = {
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

        self.all_embeddings = {
            'text-embedding-ada-002-v2':  OpenAIEmbeddings(deployment_id='text-embedding-ada-002-v2'),
            'amazon-titan-e1t-medium':  BedrockEmbeddings(deployment_id='amazon-titan-e1t-medium'),
            'gcp-textembedding-gecko-001':  GoogleEmbeddings(deployment_id='gcp-textembedding-gecko-001'),
        }

        self.open_ai_models = set([
            'text-davinci-003',
            'gpt-35-turbo',
            'gpt-4',
            'gpt-4-32k',
            'alephalpha'
        ])

        self.bedrock_models = set([
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

        self.gcp_models = set([
            'gcp-text-bison-001'
        ])

        self.huggingfacemodels = set([
            'falcon-7b',
            'falcon-40b-instruct',
            'llama2-13b-chat-hf'
        ])