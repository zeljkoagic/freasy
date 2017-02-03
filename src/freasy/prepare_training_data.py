# here we prepare nn source ranking training data

import argparse
from target_sentence import count_correct_heads
from collections import defaultdict
import dill

# argparse stuff
parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
args = parser.parse_args()

# load the target sentence pickle
target_sentences = dill.load(open("{}/pickles/{}.as_target_language.all_parses.with_{}_pos.pickle"
                                  .format(args.data_root, args.target_name, args.pos_source), "rb"))

source_rankings_for_sentences = []  # here we capture the "POS list -> source rankings" for NN training

# process each sentence
for sentence in target_sentences:
    ss_correct_for_this_sentence = defaultdict(int)
    for source_language, this_source_heads in sentence.single_source_heads.items():
        correct_heads = count_correct_heads(this_source_heads, sentence.gold_heads)
        ss_correct_for_this_sentence[source_language] = correct_heads

    source_rankings_for_sentences.append((args.target_name, sentence.idx,
                                          sentence.predicted_pos, ss_correct_for_this_sentence))

dill.dump(source_rankings_for_sentences, open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                              .format(args.data_root, args.target_name, args.pos_source), "wb"))

# TODO This should really be done with TRAIN & DEV data, not test data!!!
