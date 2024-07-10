from evaluations import ModelConfig, Document, FindBestModels, FindBestChunks, FindBestK
from score_ranker import ScoreRanker

from main import EvaluationPipeline

corpus_path = '../../langchain-converse/corpus/pdf/iep/specified/4102400943 plus Templates or Action Context Temp.pdf'

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

evaluationPipeline = EvaluationPipeline(
    corpus_path = corpus_path,
    query = query,
    truth = truth,
    max_tokens = 1200,
    temperature = 0.0,
    temperature_hf = 0.1,
    generation_count = 1
)

document = Document(corpus_path, query, truth)

find_best_models = FindBestModels(all_llms=['gpt-35-turbo'], generation_count=evaluationPipeline.model_config.generation_count, text=document.text, query=document.query, truth=document.truth)
find_best_models.run_experiment()

# use ScoreRanker to get highest scoring model
scorer = ScoreRanker()
scorer.update_scores(find_best_models.generations)
highest_scoring_models = scorer.get_highest_scoring_candidates()

# extract best model name from string
highest_scoring_model = highest_scoring_models[max(highest_scoring_models.keys())][0]
highest_scoring_model = highest_scoring_model[:highest_scoring_model.find('_')]

# use ScoreRanker to get highest scoring chunk parameters
find_best_chunks = FindBestChunks(llm=highest_scoring_model, generation_count=evaluationPipeline.model_config.generation_count, text=document.text, query=document.query, truth=document.truth)
find_best_chunks.run_experiment()

# use ScoreRanker to get highest scoring chunk parameters
scorer = ScoreRanker()
scorer.update_scores(find_best_chunks.generations)
highest_scoring_chunks = scorer.get_highest_scoring_candidates()

# extract best chunk size and overlap from string
highest_scoring_chunk = highest_scoring_chunks[max(highest_scoring_chunks.keys())][0]
best_chunk_size, best_chunk_overlap = highest_scoring_chunk.split('|')
best_chunk_size,  best_chunk_overlap = int(best_chunk_size.split('_')[1]), int(best_chunk_overlap.split('_')[1])

find_best_k = FindBestK(llm=highest_scoring_model, generation_count=evaluationPipeline.model_config.generation_count, chunk_size=best_chunk_size, chunk_overlap=best_chunk_overlap, text=document.text, query=document.query, truth=document.truth)
find_best_k.run_experiment()

print("find_best_models.generations", find_best_models.generations)
print("find_best_chunks.generations", find_best_chunks.generations)
print("find_best_k.generations", find_best_k.generations)