from rouge_score import rouge_scorer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

from langchain.document_loaders import TextLoader
from PyPDF2 import PdfReader

from helpers import get_embedding

from helpers import embeddings_with_retry, llm_with_retry

from model_manager import ModelManager

model_manager = ModelManager(max_tokens = 1200, temperature = 0.0, temperature_hf = 0.1)


class ModelConfig:
    def __init__(self, max_tokens = 1200, temperature = 0.0, temperature_hf = 0.1, generations = 3):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.temperature_hf = temperature_hf
        self.generation_count = generations



class Document:
    def __init__(self, corpus_path, query, truth):
        self.corpus_path = corpus_path
        self.query = query
        self.truth = truth
        self.text = self._extract_text()

    def _extract_text(self):
        pdf_reader = PdfReader(self.corpus_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    


class FindBestModels:
    def __init__(self, all_llms, generation_count, text, query, truth):
        self.all_llms = all_llms
        self.generation_count = generation_count
        self.text = text
        self.query = query
        self.truth = truth
        self.generations = []

    def run_experiment(self):
        for i in range(self.generation_count):
            print("going for generation", str(i))
            results = self.run_generation()
            self.generations.append(results)

    def run_generation(self):
        results = dict()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=self.text)

        # 'falcon-40b-instruct' is no longer available. see: https://github.tools.sap/AI-Playground-Projects/llm-commons/issues/128
        for llm in ['gpt-35-turbo']:
            print(llm)

            embeddings = get_embedding(llm)
            VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
            docs = VectorStore.similarity_search(query=self.query, k=5)
            chain = load_qa_chain(llm=model_manager.all_llms[llm], chain_type="stuff")

            result = 'error'
            result = llm_with_retry(docs, self.query, chain, max_retries=10)
            results[llm] = result

            scores = self.calculate_scores(result, llm)
            results.update(scores)

        return results

    def calculate_scores(self, result, llm):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(self.truth, result)
        return {
            llm + '_rouge1_precision': scores['rouge1'].precision,
            llm + '_rouge1_recall': scores['rouge1'].recall,
            llm + '_rouge1_fmeasure': scores['rouge1'].fmeasure,
            llm + '_rougeL_precision': scores['rougeL'].precision,
            llm + '_rougeL_recall': scores['rougeL'].recall,
            llm + '_rougeL_fmeasure': scores['rougeL'].fmeasure,
        }

# Usage:
# find_best_models = FindBestModels(all_llms={'falcon-40b-instruct': ..., 'gpt-35-turbo': ...}, generation_count=10, text="some text", query="some query", truth="some truth")
# find_best_models.run_experiment()



class FindBestChunks:
    def __init__(self, llm, generation_count, text, query, truth):
        self.llm = llm
        self.generation_count = generation_count
        self.text = text
        self.query = query
        self.truth = truth
        self.generations = []
        self.chunk_sizes = [200]#, 400, 600, 800, 1000, 1200, 1400]
        self.chunk_overlaps = [20]#, 40, 60, 100, 150, 200, 300, 400, 500, 1000]

    def run_experiment(self):
        for i in range(self.generation_count):
            print("going for generation", str(i))
            results = self.run_generation()
            self.generations.append(results)

    def run_generation(self):
        results = dict()
        for chunk_size in self.chunk_sizes:
            for chunk_overlap in self.chunk_overlaps:
                if chunk_size < chunk_overlap:
                    continue

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=self.text)

                params = "cs_" + str(chunk_size) + "|co_" + str(chunk_overlap)
                print(params)

                embeddings = get_embedding(self.llm)
                VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
                docs = VectorStore.similarity_search(query=self.query, k=5)
                print("self.llm is", self.llm)
                chain = load_qa_chain(llm=model_manager.all_llms[self.llm], chain_type="stuff")

                result = 'error'
                result = llm_with_retry(docs, self.query, chain, max_retries=10)
                results[params] = result

                scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                scores = scorer.score(self.truth, result)

                results[params + '_rouge1_precision'] = scores['rouge1'].precision
                results[params + '_rouge1_recall'] = scores['rouge1'].recall
                results[params + '_rouge1_fmeasure'] = scores['rouge1'].fmeasure
                results[params + '_rougeL_precision'] = scores['rougeL'].precision
                results[params + '_rougeL_recall'] = scores['rougeL'].recall
                results[params + '_rougeL_fmeasure'] = scores['rougeL'].fmeasure

        return results

# Usage:
# find_best_chunks = FindBestChunks(llm=all_llms['gpt-35-turbo'], generation_count=10, text="some text", query="some query", truth="some truth")
# find_best_chunks.run_experiment()



class FindBestK:
    def __init__(self, llm, generation_count, chunk_size, chunk_overlap, text, query, truth):
        self.llm = llm
        self.generation_count = generation_count
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text = text
        self.query = query
        self.truth = truth
        self.generations = []
        self.ks = list(range(1, 2))

    def run_experiment(self):
        for i in range(self.generation_count):
            print("going for generation", str(i))
            results = self.run_generation()
            self.generations.append(results)

    def run_generation(self):
        results = dict()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text=self.text)

        for k in self.ks:
            result = self.process_k(chunks, k)
            results[str(k)] = result
        return results

    def process_k(self, chunks, k):
        embeddings = get_embedding(self.llm)
        VectorStore = embeddings_with_retry(chunks, embeddings, max_retries=10)
        docs = VectorStore.similarity_search(query=self.query, k=k)
        chain = load_qa_chain(llm=model_manager.all_llms[self.llm], chain_type="stuff")

        result = 'error'
        result = llm_with_retry(docs, self.query, chain, max_retries=10)

        scores = self.calculate_scores(result)
        return scores

    def calculate_scores(self, result):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(self.truth, result)
        return {
            'rouge1_precision': scores['rouge1'].precision,
            'rouge1_recall': scores['rouge1'].recall,
            'rouge1_fmeasure': scores['rouge1'].fmeasure,
            'rougeL_precision': scores['rougeL'].precision,
            'rougeL_recall': scores['rougeL'].recall,
            'rougeL_fmeasure': scores['rougeL'].fmeasure,
        }

# Usage:
# find_best_k = FindBestK(llm='falcon-40b-instruct', generation_count=10, text="some text", query="some query", truth="some truth")
# find_best_k.run_experiment()