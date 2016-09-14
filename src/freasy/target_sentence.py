class Arc(object):
    def __init__(self, lang, source, target, deprel, pos_source, weight=1.0):
        self.lang = lang
        self.source = source
        self.target = target
        self.deprel = deprel
        self.weight = weight
        self.pos_source = pos_source

    def __repr__(self):
        return "{}:{}->{}|{}|{}|{}".format(self.lang, self.source, self.target, self.deprel, self.weight,
                                           self.pos_source)

    def __eq__(self, other):
        if self.lang == other.lang and self.source == other.source and self.target == other.target and \
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

    #def __str__(self):
    #    return "{}\n{}\n{}\n{}".format(self.tokens, self.gold_pos, self.pred_pos, self.proj_pos)
