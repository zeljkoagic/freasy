# open target pickle
# take a sentence
# create a cle-input matrix from all the arcs
# decode, evaluate
# pip install git+https://github.com/andersjo/dependency_decoding.git

import dill
import argparse
from target_sentence import count_correct_heads
from collections import defaultdict
import numpy as np
from dependency_decoding import chu_liu_edmonds
from softmax import invert, softmax


def create_ss_tensor(n, single_source_heads):
    tensor = np.zeros((n+1, n+1, len(single_source_heads)), dtype=float)
    language_sequence = []
    for lang_idx, (language, heads) in enumerate(single_source_heads.items()):
        language_sequence.append(language)
        for j, head in enumerate(heads):
            tensor[j+1, head, lang_idx] = 1.0
    return tensor, language_sequence

# argparse stuff
parser = argparse.ArgumentParser(description="Performs language weighting experiments.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
parser.add_argument("--granularity", required=True, help="target language estimation granularity", type=int)
# parser.add_argument('--use_softmax', required=True, choices=[0, 1], help="smooth source contributions?", type=int)
parser.add_argument("--temperature", required=True, help="softmax temperature", type=float)
parser.add_argument("--weighting_method", required=True, choices=["klcpos3", "wals", "langid"], help="source weighting")
args = parser.parse_args()

# load the target sentence pickle
target_sentences = dill.load(open("{}/pickles/{}.as_target_language.all_parses.with_{}_pos.pickle"
                                  .format(args.data_root, args.target_name, args.pos_source), "rb"))
# load the language mappings
source_weights = dill.load(open("{}/pickles/{}.source_language_mappings.with_{}_pos.pickle"
                                .format(args.data_root, args.target_name, args.pos_source), "rb"))

ss_correct = defaultdict(int)  # record single source performance

ss_predicted_correct = 0
ss_predicted_sources_counter = defaultdict(int)  # for counting the contributing sources

ss_voted_unweighted_correct = 0
ss_voted_weighted_correct = 0
ms_correct = 0
total = 0

# process each sentence
for sentence in target_sentences:

    total += len(sentence.tokens)

    # read the predicted best single-source parser, and the source weights
    predicted_best_single_source, source_distribution = \
        source_weights[args.weighting_method][args.granularity][sentence.idx]

    ss_predicted_sources_counter[predicted_best_single_source] += 1

    for source_language, this_source_heads in sentence.single_source_heads.items():

        correct_heads = count_correct_heads(this_source_heads, sentence.gold_heads)
        ss_correct[source_language] += correct_heads  # record scores of single-source parsers

        # collect score for the predicted best single-source parser
        if source_language == predicted_best_single_source:
            ss_predicted_correct += correct_heads

    ms_correct += count_correct_heads(sentence.multi_source_heads, sentence.gold_heads)

    # get the single-source parses tensor, and the source language ordering
    # the ordering is important because tensor axis 2 is numerically indexed
    ss_tensor, ss_ordering = create_ss_tensor(len(sentence.tokens), sentence.single_source_heads)

    # vote and decode without weights
    ss_matrix_voted_unweighted = np.sum(ss_tensor, axis=2)
    ss_voted_unweighted_heads, _ = chu_liu_edmonds(ss_matrix_voted_unweighted)
    ss_voted_unweighted_correct += count_correct_heads(ss_voted_unweighted_heads[1:], sentence.gold_heads)

    # vote and decode with weighting
    source_distribution = invert(softmax(source_distribution, args.temperature))

    for idx, source_language in enumerate(ss_ordering):
        weight = source_distribution[source_language]
        ss_tensor[:, :, idx] *= weight

    ss_matrix_voted_weighted = np.sum(ss_tensor, axis=2)
    ss_voted_weighted_heads, _ = chu_liu_edmonds(ss_matrix_voted_weighted)
    ss_voted_weighted_correct += count_correct_heads(ss_voted_weighted_heads[1:], sentence.gold_heads)

# extract the REAL best single source
true_best_single_source = None
max_correct = -1
for source_language, correct_heads in ss_correct.items():
    if correct_heads > max_correct:
        true_best_single_source = source_language
        max_correct = correct_heads

print(true_best_single_source, "{0:.2f}".format((ss_correct[true_best_single_source]/total)*100))
print("ss predicted: {0:.2f}".format((ss_predicted_correct/total)*100), ss_oracle_sources_counter)
print("ms: {0:.2f}".format((ms_correct/total)*100))
print("vote w=1: {0:.2f}".format((ss_voted_unweighted_correct/total)*100))
print("vote w=x: {0:.2f}".format((ss_voted_weighted_correct/total)*100))

# TODO Do we need a sentence-level best gold single-source score?
