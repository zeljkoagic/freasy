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


def create_ss_tensor(n, single_source_heads):
    tensor = np.zeros((n+1, n+1, len(single_source_heads)), dtype=float)
    language_sequence = []
    for lang_idx, (language, heads) in enumerate(single_source_heads.items()):
        language_sequence.append(language)
        for j, head in enumerate(heads):
            tensor[j+1, head-1, lang_idx] = 1.0
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
ms_correct = 0
total = 0

# process each sentence
for sentence in target_sentences:

    total += len(sentence.tokens)

    # read the predicted best single-source parser, and the source weights
    predicted_best_single_source, source_distribution = \
        source_weights[args.weighting_method][args.granularity][sentence.idx]

    for source_language, this_source_heads in sentence.single_source_heads.items():

        correct_heads = count_correct_heads(this_source_heads, sentence.gold_heads)
        ss_correct[source_language] += correct_heads  # record scores of single-source parsers

        # collect score for the predicted best single-source parser
        if source_language == predicted_best_single_source:
            ss_predicted_correct += correct_heads

    ms_correct += count_correct_heads(sentence.multi_source_heads, sentence.gold_heads)

    # decode the voted, with or without weights for the given weighting method
    ss_tensor, ss_ordering = create_ss_tensor(len(sentence.tokens), sentence.single_source_heads)
    print(sentence.gold_heads)
    print(ss_tensor[:, :, 0])

# extract the REAL best single source
true_best_single_source = None
max_correct = -1
for source_language, correct_heads in ss_correct.items():
    if correct_heads > max_correct:
        true_best_single_source = source_language
        max_correct = correct_heads

print(true_best_single_source, "{0:.2f}".format((ss_correct[true_best_single_source]/total)*100))
print("{0:.2f}".format((ss_predicted_correct/total)*100))
print("{0:.2f}".format((ms_correct/total)*100))