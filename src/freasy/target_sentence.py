class TargetSentence:
    """
    Target sentence with all the parses coming from various sources.
    """
    def __init__(self, idx, language, tokens, gold_heads, gold_pos, predicted_pos,
                 multi_source_heads, single_source_heads, pos_source_type):
        """
        :param idx: Sentence ID in the test set
        :param language: True language name
        :param tokens: List of sentence tokens
        :param gold_heads: List of heads for each sentence token (CoNLL style)
        :param gold_pos: List of gold POS tags
        :param predicted_pos: List of predicted POS tags
        :param multi_source_heads: List of heads predicted by the multi-source delexicalized parser ("ALL")
        :param multi_source_heads: Dictionary of head lists for different single-source parsers
        :param pos_source_type: Where does the POS come from, gold-proj-pred
        """
        self.idx = idx
        self.language = language
        self.tokens = tokens
        self.gold_heads = gold_heads
        self.gold_pos = gold_pos
        self.predicted_pos = predicted_pos
        self.multi_source_heads = multi_source_heads
        self.single_source_heads = single_source_heads
        self.pos_source_type = pos_source_type


def count_correct_heads(heads_predicted, heads_gold):
    return sum([int(predicted == gold) for predicted, gold in zip(heads_predicted, heads_gold)])
