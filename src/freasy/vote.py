# open target pickle
# take a sentence
# create a cle-input matrix from all the arcs
# decode, evaluate

# pip install git+https://github.com/andersjo/dependency_decoding.git

import sys
import pickle
import numpy as np
from target_sentence import Arc, TargetSentence
from dependency_decoding import chu_liu_edmonds
import dill
from collections import defaultdict
import math

# FIXME Maybe better to softmax entire matrix?
from softmax import softmax


def load_tensor(n, arcs):
    """
    Creates a tensor from a list of dependency arcs.
    """
    # 3-dimensional tensor: [dependent, head, source_language]
    # TODO Currently does not support labels.
    tensor = np.zeros((n+1, n+1, len(arcs)), dtype=float)
    sources = []

    # iterate through arcs
    for lang_index, (lang, lang_arcs) in enumerate(arcs.items()):
        sources.append(lang)  # store the languages in a particular order

        for arc in lang_arcs["gold"]:
            tensor[arc.dependent, arc.head, lang_index] = arc.weight  # fill the tensor with weights

    return tensor, sources

# load the target sentence pickle
target_sentences = dill.load(open(sys.argv[1], "rb"))

# FIXME load the source weights
source_weights = dill.load(open(sys.argv[2], "rb"))

correct = defaultdict(int)
total = defaultdict(int)

# process each sentence
for sentence in target_sentences:
    tensor, sources = load_tensor(len(sentence.tokens), sentence.arcs_from_sources)

    # here we decode for the individual sources
    # TODO This decoding is trivial because individual slices are already trees!
    # source order is important because the tensor is not explicitly indexed by source names
    for idx, source in enumerate(sources):

        # FIXME This will not work, we need to index by source!!!
        if source != "ALL":
            tensor[:, :, idx] *= math.pow(1 / source_weights["wals"][100][sentence.idx][1][source], 10)

        heads, _ = chu_liu_edmonds(tensor[:, :, idx])
        heads = heads[1:]

        correct[source] += sum([predicted == gold for predicted, gold in zip(heads, [arc.head for arc in sentence.gold_arcs])])
        total[source] += len(sentence.tokens)

    # this is where voting happens, currently all weights are 1.0
    voted = np.sum(tensor, axis=2)

    heads, _ = chu_liu_edmonds(voted)
    heads = heads[1:]

    correct["voted"] += sum([predicted == gold for predicted, gold in zip(heads, [arc.head for arc in sentence.gold_arcs])])
    total["voted"] += len(sentence.tokens)


for source, corr in correct.items():
    print(source, corr/total[source])

