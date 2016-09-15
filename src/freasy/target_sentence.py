class Arc(object):
    """
    Dependency edges between tokens.
    """
    def __init__(self, lang, head, dependent, deprel, pos_source, weight=1.0):
        self.lang = lang
        self.head = head
        self.dependent = dependent
        self.deprel = deprel
        self.weight = weight
        self.pos_source = pos_source

    def __repr__(self):
        return "{}:{}->{}|{}|{}|{}".format(self.lang, self.head, self.dependent, self.deprel, self.weight,
                                           self.pos_source)

    def __eq__(self, other):
        if self.lang == other.lang and self.head == other.head and self.dependent == other.dependent and \
                        self.weight == other.weight and self.pos_source == other.pos_source:
            return True
        return False


class TargetSentence:
    """
    Target sentence with all the parses coming from various sources.
    """
    def __init__(self, idx, lang, tokens, gold_arcs, pos, arcs_from_sources):
        self.idx = idx
        self.lang = lang
        self.tokens = tokens
        self.gold_arcs = gold_arcs
        self.pos = pos
        self.arcs_from_sources = arcs_from_sources

# TODO Should a target sentence have different calculate() methods for LAS, UAS, POS accuracy, etc.?
