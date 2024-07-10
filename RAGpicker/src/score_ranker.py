class ScoreRanker:
    def __init__(self):
        self.rouge1_precision_scores = {}
        self.rouge1_recall_scores = {}
        self.rouge1_fmeasure_scores = {}
        self.rougeL_precision_scores = {}
        self.rougeL_recall_scores = {}
        self.rougeL_fmeasure_scores = {}

        self.all_scoring = [self.rouge1_fmeasure_scores, self.rougeL_fmeasure_scores]
        self.all_scoring_str = ['rouge1_fmeasure', 'rougeL_fmeasure']

    def update_scores(self, generations):
        for generation in generations:
            for column in generation:
                for i in range(len(self.all_scoring)):
                    if self.all_scoring_str[i] in column:
                        score = generation[column]
                        if score in self.all_scoring[i]:
                            self.all_scoring[i][score].append(column)
                        else:
                            self.all_scoring[i][score] = [column]

    def print_scores(self):
        for scoring in self.all_scoring:
            sorted_scores = sorted(scoring)[::-1]
            for i in range(1, 10):
                print(scoring[sorted_scores[i]], sorted_scores[i])
            print()

    def get_highest_scoring_candidates(self):
        highest_scoring_candidates = {}
        for scoring in self.all_scoring:
            print("scoring is", scoring)
            sorted_scores = sorted(scoring)[::-1]
            highest_scoring_candidates[max(scoring.keys())] = scoring[max(scoring.keys())]

        return highest_scoring_candidates


# scorer = ScoreRanker()
# scorer.update_scores(generations = [])
# scorer.print_scores()


# highest_scoring_candidates = scorer.get_highest_scoring_candidates()
# print(highest_scoring_candidates)  # Outputs: Highest scoring candidates