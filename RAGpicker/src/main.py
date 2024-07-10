from evaluations import ModelConfig, Document, FindBestModels, FindBestChunks, FindBestK
from score_ranker import ScoreRanker


class EvaluationPipeline:
    def __init__(self, corpus_path, query, truth, max_tokens, temperature, temperature_hf, generation_count):

        self.model_config = ModelConfig(max_tokens, temperature, temperature_hf, generation_count)

        self.document = Document(corpus_path, query, truth)



def main():
    
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
        generation_count = 3
    )

    document = Document(corpus_path, query, truth)

    find_best_models = FindBestModels(all_llms=['falcon-40b-instruct', 'gpt-35-turbo'], generation_count=evaluationPipeline.model_config.generation_count, text=document.text, query=document.query, truth=document.truth)
    find_best_models.run_experiment()

    # use ScoreRanker to get highest scoring model
    scorer = ScoreRanker()
    scorer.update_scores(find_best_models.generations)
    highest_scoring_models = scorer.get_highest_scoring_candidates()

    find_best_chunks = FindBestChunks(llm=highest_scoring_models, generation_count=evaluationPipeline.model_config.generation_count, text=document.text, query=document.query, truth=document.truth)
    find_best_chunks.run_experiment()

    # use ScoreRanker to get highest scoring chunk parameters
    scorer = ScoreRanker()
    scorer.update_scores(find_best_chunks.generations)
    highest_scoring_chunks = scorer.get_highest_scoring_candidates()

    find_best_k = FindBestK(llm=highest_scoring_models, generation_count=evaluationPipeline.model_config.generation_count, text=document.text, query=document.query, truth=document.truth)
    find_best_k.run_experiment()

    # use ScoreRanker to get highest scoring k parameters
    scorer = ScoreRanker()
    scorer.update_scores(find_best_k.generations)
    highest_scoring_k = scorer.get_highest_scoring_candidates()
