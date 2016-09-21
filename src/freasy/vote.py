# open target pickle
# take a sentence
# create a cle-input matrix from all the arcs
# decode, evaluate
# pip install git+https://github.com/andersjo/dependency_decoding.git

import dill
import argparse
from target_sentence import count_correct_heads
from collections import defaultdict

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
total = 0

# process each sentence
for sentence in target_sentences:

    total += len(sentence.tokens)

    # find the REAL best single source
    for source_language, this_source_heads in sentence.single_source_heads.items():
        ss_correct += count_correct_heads(this_source_heads, sentence.gold_heads)

    # also the PREDICTED best single source
    # evaluate the multi-source
    # decode the voted, with or without weights for the given weighting method

print("%.2f".format((ss_correct["cs"]/total)*100))
