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
import argparse

# FIXME Maybe better to softmax entire matrix? Softmax makes sense after the language weights are applied.
from softmax import softmax, invert


def load_tensor(n, arcs, pos_source):
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

        for arc in lang_arcs[pos_source]:
            tensor[arc.dependent, arc.head, lang_index] = arc.weight  # fill the tensor with weights

    return tensor, sources

# argparse stuff
parser = argparse.ArgumentParser(description="Performs language weighting experiments.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
parser.add_argument("--weighting_method", required=True, choices=["klcpos3", "wals", "langid"],
                    help="source weighting method")
parser.add_argument("--granularity", required=True, help="target language estimation granularity", type=int)
parser.add_argument('--use_softmax', required=True, choices=[0, 1],
                    help="use softmax to smooth source contributions?", type=int)
parser.add_argument("--temperature", required=False, help="softmax temperature", type=float)
args = parser.parse_args()

# load the target sentence pickle
target_sentences = dill.load(open("{}/pickles/{}.as_target_language.all_parses.pickle"
                                  .format(args.data_root, args.target_name), "rb"))

# load the source weights
source_weights = dill.load(open("{}/pickles/{}.source_language_mappings.with_{}_pos.pickle"
                                .format(args.data_root, args.target_name, args.pos_source), "rb"))

# TODO Isn't this now redundant?
source_weights_for_method_and_granularity = source_weights[args.weighting_method][args.granularity]

assert args.granularity in source_weights[args.weighting_method], \
    "You must choose one of these as granularity: %s" % source_weights[args.weighting_method].keys()

assert bool(args.use_softmax) == bool(args.temperature), "If args.softmax, then args.temperature as well!"

correct = defaultdict(int)
total = defaultdict(int)

# process each sentence
for sentence in target_sentences:

    # create tensor from arcs
    tensor, sources = load_tensor(len(sentence.tokens), sentence.arcs_from_sources, args.pos_source)

    # extract the weights
    best_source_for_sentence, source_weights_for_sentence = source_weights_for_method_and_granularity[sentence.idx]

    # apply softmax
    if args.use_softmax:
        source_weights_for_sentence = invert(softmax(sources_distribution=source_weights_for_sentence,
                                                     temperature=args.temperature))

    print(source_weights_for_sentence)

    # here we decode for the individual sources
    # source order is important because the tensor is not explicitly indexed by source names
    for idx, source in enumerate(sources):

        # TODO Individual slices are already trees! Makes sense only to decode for voted.
        heads, _ = chu_liu_edmonds(tensor[:, :, idx])
        heads = heads[1:]

        correct[source] += sum([predicted == gold for predicted, gold
                                in zip(heads, [arc.head for arc in sentence.gold_arcs])])
        total[source] += len(sentence.tokens)

        # apply weights TODO is this the right place to do it?
        if source != "ALL" and args.use_softmax:
            tensor[:, :, idx] *= source_weights_for_sentence[source]

    # this is where voting happens, currently all weights are 1.0
    voted = np.sum(tensor, axis=2)

    heads, _ = chu_liu_edmonds(voted)
    heads = heads[1:]

    correct["voted"] += sum([predicted == gold for predicted, gold
                             in zip(heads, [arc.head for arc in sentence.gold_arcs])])
    total["voted"] += len(sentence.tokens)


for source, corr in correct.items():
    print("%.2f\t%s" % ((corr/total[source])*100, source))

