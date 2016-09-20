# open target pickle
# take a sentence
# create a cle-input matrix from all the arcs
# decode, evaluate
# pip install git+https://github.com/andersjo/dependency_decoding.git

import numpy as np
from dependency_decoding import chu_liu_edmonds
import dill
from collections import defaultdict
import argparse
from softmax import softmax, invert


def load_tensor(n, arcs, pos_source):
    """
    Creates a tensor from a list of dependency arcs.
    """
    # 3-dimensional tensor: [dependent, head, source_language]
    # TODO Currently does not support labels.
    single_source_tensor = np.zeros((n+1, n+1, len(arcs)), dtype=float)
    sources = []
    multi_source_matrix = np.zeros((n+1, n+1))

    # iterate through arcs
    for lang_index, (lang, lang_arcs) in enumerate(arcs.items()):
        # we do this for all single languages
        if lang != "ALL":
            sources.append(lang)  # store the languages in a particular order
            for arc in lang_arcs[pos_source]:
                single_source_tensor[arc.dependent, arc.head, lang_index] = arc.weight  # fill the tensor with weights
        else:
            for arc in lang_arcs[pos_source]:
                # multisource gets special treatment
                multi_source_matrix[arc.dependent, arc.head] = arc.weight

    return single_source_tensor, single_source_tensor, multi_source_matrix, sources


def get_matrix(n, arcs):
    matrix = np.zeros((n+1, n+1))
    for arc in arcs:
        matrix[arc.dependent, arc.head] = arc.weight
    return matrix


def get_heads(matrix):
    """
    Converts matrix representation of dependency tree to heads list.
    """
    heads = np.zeros(matrix.shape[0])

    for dependent, row in enumerate(matrix):
        for head in row:
            if head != 0:
                heads[dependent] = head

    return heads.tolist()[1:]


def count_correct(heads_predicted, heads_gold):
    """
    Counts number of correct heads.
    """
    return sum([int(predicted == gold) for predicted, gold in zip(heads_predicted, heads_gold)])


# argparse stuff
parser = argparse.ArgumentParser(description="Performs language weighting experiments.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
#parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
#parser.add_argument("--weighting_method", required=True, choices=["klcpos3", "wals", "langid"],
#                    help="source weighting method")
#parser.add_argument("--granularity", required=True, help="target language estimation granularity", type=int)
#parser.add_argument('--use_softmax', required=True, choices=[0, 1],
#                    help="use softmax to smooth source contributions?", type=int)
#parser.add_argument("--temperature", required=False, help="softmax temperature", type=float)
args = parser.parse_args()

# load the target sentence pickle
target_sentences = dill.load(open("{}/pickles/{}.as_target_language.all_parses.pickle"
                                  .format(args.data_root, args.target_name), "rb"))

#assert args.granularity in source_weights[args.weighting_method], \
#    "You must choose one of these as granularity: %s" % source_weights[args.weighting_method].keys()

#if args.use_softmax:
#    assert args.temperature, "If args.softmax, then args.temperature as well!"

weighting_methods = ["wals"]
pos_sources = ["proj"]
granularities = [100]
temperatures = [0.2]

source_weights = defaultdict()
for pos_source in pos_sources:
    source_weights[pos_source] = dill.load(open("{}/pickles/{}.source_language_mappings.with_{}_pos.pickle"
                                                .format(args.data_root, args.target_name, pos_source), "rb"))

correct_ss_estimated = 0
correct_ss_true = 0
correct_ms = 0
correct_voted_weighted = 0
correct_voted = 0
total = 0

# process each sentence
for sentence in target_sentences:

    heads_gold = get_heads(get_matrix(len(sentence.tokens), sentence.gold_arcs))
    print(get_matrix(sentence.gold_arcs))
    print(sentence.tokens)
    print(heads_gold)

    for pos_source in pos_sources:
        for weighting_method in weighting_methods:
            for granularity in granularities:
                for temperature in temperatures:

                    # TODO Find *true* best single source---WHICH GRANULARITY?

                    # create tensor from arcs
                    ss_tensor, ss_tensor_weighted, ms_matrix, sources = \
                        load_tensor(len(sentence.tokens), sentence.arcs_from_sources, pos_source)

                    heads_ms = get_heads(ms_matrix)
                    correct_ms += count_correct(heads_ms, heads_gold)

                    # get the weighting results
                    estimated_best_source_for_sentence, source_weights_for_sentence = \
                        source_weights[pos_source][weighting_method][granularity][sentence.idx]

                    # apply softmax, temperature, and inverse
                    source_weights_for_sentence = invert(softmax(sources_distribution=source_weights_for_sentence,
                                                                 temperature=temperature))

                    # here we decode for the individual sources
                    # source order is important because the tensor is not explicitly indexed by source names
                    for idx, source in enumerate(sources):

                        # get the best source heads
                        if source == estimated_best_source_for_sentence:
                            heads_ss = get_heads(ss_tensor[:, :, idx])
                            correct_ss_estimated += count_correct(heads_ss, heads_gold)

                        # apply weights
                        ss_tensor_weighted[:, :, idx] *= source_weights_for_sentence[source]

                    # weighted voting
                    matrix_voted = np.sum(ss_tensor, axis=2)  # TODO Should be done only once for this one
                    matrix_voted_weighted = np.sum(ss_tensor_weighted, axis=2)

                    heads_voted, _ = chu_liu_edmonds(matrix_voted)
                    heads_voted = heads_voted[1:]

                    heads_voted_weighted, _ = chu_liu_edmonds(matrix_voted_weighted)
                    heads_voted_weighted = heads_voted_weighted[1:]

                    correct_voted += count_correct(heads_voted, heads_gold)
                    correct_voted_weighted += count_correct(heads_voted_weighted, heads_gold)

                    total += len(sentence.tokens)

print(correct_ss_estimated/total)
print(correct_ms/total)
print(correct_voted/total)
print(correct_voted_weighted/total)
