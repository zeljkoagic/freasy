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
import operator
from scipy.stats import kendalltau, spearmanr


def create_ss_tensor(n, single_source_heads):
    tensor = np.zeros((n+1, n+1, len(single_source_heads)), dtype=float)
    language_sequence = []
    for lang_idx, (language, heads) in enumerate(single_source_heads.items()):
        language_sequence.append(language)
        for j, head in enumerate(heads):
            tensor[j+1, head, lang_idx] = 1.0
    return tensor, language_sequence


def average_precision_at_n(system, gold):
    precisions_at_k = []
    for k in range(1, len(system)+1):
        p_at_k = sum([int(s == g) for s, g in zip(system[0:k], gold[0:k])])
        p_at_k /= k
        precisions_at_k.append(p_at_k)
    print(system)
    print(gold)
    return sum(precisions_at_k) / len(system)


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

ss_oracle_correct = 0  # always choose best source parse for each sentence!
ss_oracle_sources_counter = defaultdict(int)  # its corresponding counter
ss_oracle_sources_counter_with_ties = defaultdict(int)

ss_predicted_correct = 0  # accuracy of source weighting-based best single source detection
ss_predicted_sources_counter = defaultdict(int)  # for counting the contributing sources

ss_voted_unweighted_correct = 0
ss_voted_weighted_correct = 0
ms_correct = 0
total = 0

correct_pos = 0

predicted_source_rankings = []  # to store lists of source rankings for later averaging

# process each sentence
for sentence in target_sentences:

    total += len(sentence.tokens)

    # read the predicted best single-source parser, and the source weights
    predicted_best_single_source, source_distribution = \
        source_weights[args.weighting_method][args.granularity][sentence.idx]

    ss_predicted_sources_counter[predicted_best_single_source] += 1

    #print(sentence.tokens, predicted_best_single_source, source_distribution)

    # capture the best source for this sentence!
    true_best_single_source = None
    max_correct = -1

    for source_language, this_source_heads in sentence.single_source_heads.items():

        correct_heads = count_correct_heads(this_source_heads, sentence.gold_heads)
        ss_correct[source_language] += correct_heads  # record scores of single-source parsers

        # TODO There are multiple languages with equal scores here, thus the non-determinism in the output.
        if correct_heads > max_correct:
            true_best_single_source = source_language
            max_correct = correct_heads

        # collect score for the predicted best single-source parser
        if source_language == predicted_best_single_source:
            ss_predicted_correct += correct_heads

    # for each target sentence, we always pick the best source
    ss_oracle_correct += \
        count_correct_heads(sentence.single_source_heads[true_best_single_source], sentence.gold_heads)
    ss_oracle_sources_counter[true_best_single_source] += 1

    # same as above, but...
    # find ALL languages that score best score for sentence, count them
    all_single_sources = sorted(ss_correct.items(), key=operator.itemgetter(1), reverse=True)
    all_best_single_sources = []
    best_score = all_single_sources[0][1]  # the first item has highest score
    for src, cnt in all_single_sources:
        if cnt == best_score:
            all_best_single_sources.append(src)
    for best_single_source in all_best_single_sources:
        ss_oracle_sources_counter_with_ties[best_single_source] += 1

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

    # get the sorted source language list, i.e., source ranking
    sorted_source_distribution = sorted(source_distribution.items(), key=operator.itemgetter(1), reverse=True)
    predicted_source_rankings.append([l for l, _ in sorted_source_distribution])

    for idx, source_language in enumerate(ss_ordering):
        weight = source_distribution[source_language]
        ss_tensor[:, :, idx] *= weight

    ss_matrix_voted_weighted = np.sum(ss_tensor, axis=2)
    ss_voted_weighted_heads, _ = chu_liu_edmonds(ss_matrix_voted_weighted)
    ss_voted_weighted_correct += count_correct_heads(ss_voted_weighted_heads[1:], sentence.gold_heads)

    correct_pos += count_correct_heads(sentence.predicted_pos, sentence.gold_pos)

# extract the REAL best single source TODO This is macro!
true_best_single_source = None
max_correct = -1

for source_language, correct_heads in ss_correct.items():
    if correct_heads > max_correct:
        true_best_single_source = source_language
        max_correct = correct_heads

true_source_ranking = sorted(ss_correct.items(), key=operator.itemgetter(1), reverse=True)
true_source_ranking = [l for l, _ in true_source_ranking]

avg_kt = 0  # for kendall's tau
avg_sr = 0  # for spearman's rho
avg_rank_of_first_pick = 0  # TODO

# map the language names to numeric ranks
lang_to_rank_mapping_gold = dict(zip(true_source_ranking, range(1, len(true_source_ranking)+1)))
gold_ranking = list(lang_to_rank_mapping_gold.values())

for ranking in predicted_source_rankings:

    # map the language names to numeric ranks
    lang_to_rank_mapping_system = dict(zip(ranking, range(1, len(ranking)+1)))
    system_ranking = list(lang_to_rank_mapping_system.values())

    predicted_best_source = ranking[0]
    predicted_best_source_rank_in_gold = lang_to_rank_mapping_gold[predicted_best_source]
    avg_rank_of_first_pick += predicted_best_source_rank_in_gold

    t, _ = kendalltau(system_ranking, gold_ranking, nan_policy="omit")
    r, _ = spearmanr(system_ranking, gold_ranking, nan_policy="omit")

    avg_kt += t
    avg_sr += r

avg_kt /= len(predicted_source_rankings)
avg_sr /= len(predicted_source_rankings)
avg_rank_of_first_pick /= len(predicted_source_rankings)

print("kendall tau_b, spearman r, average rank of first choice: ", avg_kt, avg_sr, avg_rank_of_first_pick)

print("true best ss: ", true_best_single_source, "{0:.2f}".format((ss_correct[true_best_single_source]/total)*100))

print("ss per-sentence oracle: {0:.2f}".format((ss_oracle_correct/total)*100))
print("ss predicted: {0:.2f}".format((ss_predicted_correct/total)*100))

print("ms: {0:.2f}".format((ms_correct/total)*100))

print("vote w=1: {0:.2f}".format((ss_voted_unweighted_correct/total)*100))
print("vote w=x: {0:.2f}".format((ss_voted_weighted_correct/total)*100))
print("pos acc: {0:.2f}".format((correct_pos/total)*100))

print(ss_oracle_sources_counter_with_ties)