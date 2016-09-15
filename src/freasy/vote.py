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


def load_tensor(n, arcs):
    tensor = np.zeros((n+1, n+1, len(arcs)), dtype=float)
    sources = []

    for lang_index, (lang, lang_arcs) in enumerate(arcs.items()):
        sources.append(lang)  # store the languages in a particular order

        for arc in lang_arcs["pred"]:
            tensor[arc.target, arc.source, lang_index] = arc.weight  # fill the tensor with weights

    return tensor, sources

target_sentences = dill.load(open(sys.argv[1], "rb"))

correct = defaultdict(int)
total = defaultdict(int)

for sentence in target_sentences:
    tensor, sources = load_tensor(len(sentence.tokens), sentence.arcs_from_sources)

    for idx, source in enumerate(sources):

        heads, _ = chu_liu_edmonds(tensor[:, :, idx])
        heads = heads[1:]

        correct[source] += sum([predicted == gold for predicted, gold in zip(heads, [arc.head for arc in sentence.gold_arcs])])
        total[source] += len(sentence.tokens)

    voted = np.sum(tensor, axis=2)

    heads, _ = chu_liu_edmonds(voted)
    heads = heads[1:]

    correct["ALL"] += sum([predicted == gold for predicted, gold in zip(heads, [arc.head for arc in sentence.gold_arcs])])
    total["ALL"] += len(sentence.tokens)


for source, corr in correct.items():
    print(source, corr/total[source])

