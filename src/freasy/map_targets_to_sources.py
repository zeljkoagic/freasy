# for each target language test set
# calculate all mappings: klcpos3, langid, wals
# store these structures as pickles to load in decoding

# load all training sets
# for each test set target sentence
# estimate mappings

# pip install git+https://github.com/saffsd/langid.py.git

import argparse
import sys
from collections import defaultdict
from langid.langid import LanguageIdentifier, model
from functools import partial
import wals
import klcpos3
from softmax import softmax, invert
import dill


def read_sentences(filename):
    sentences = []
    current = [("###END###", "###END###"), ("###START###", "###START###")]
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line:
                idx, token, _, _, tag, _ = line.split("\t", 5)  # conll 2009
                current.append((token, tag))
            else:
                current.append(("###END###", "###END###"))
                current.append(("###START###", "###START###"))
                sentences.append(current)
                current = [("###END###", "###END###"), ("###START###", "###START###")]
    return sentences


def get_distributions_from_langid(target_sentence_sample, identifier, list_of_source_languages=None):
    # get all tokens from the sample in a list
    target_tokens = [token for target_sentence in target_sentence_sample for token, _ in target_sentence]
    target_string = " ".join(target_tokens)

    best_source = identifier.classify(target_string)[0]
    distribution_of_sources = invert(dict(identifier.rank(target_string)))

    return best_source, distribution_of_sources


# argparse stuff
parser = argparse.ArgumentParser(description="Predicts appropriate sources for a given target language.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--wals_data", required=True, help="path to WALS database CSV")
parser.add_argument("--iso_mappings", required=True, help="path to ISO 639-2/3 mappings file")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
args = parser.parse_args()

# list of all languages used in the experiment
all_languages = sorted(["ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "he", "hi", "hr", "hu",
                        "id", "it", "nl", "no", "pl", "pt", "ro", "sl", "sv", "ta"])

# target language name comes in from the console, to make it target-centric for parallel
target_language = args.target_name

# get the list of source language names for convenience
source_languages = sorted(set(all_languages) - {target_language})

path_to_data_files = args.data_root  # path to all data files

# results are stored here: all_mappings[METHOD][GRANULARITY][target_sentence_id] = (best_source, sources_distribution)
all_mappings = defaultdict(lambda: defaultdict(lambda: defaultdict(tuple)))

# get all trigram frequencies for klcpos3
trigram_freqs_for_sources = defaultdict()

# initialize langid
identifier_raw = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier_raw.set_languages(source_languages)  # only sources here, raw langid method
identifier_wals = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier_wals.set_languages(all_languages)

wals_data = wals.read_wals_csv(args.wals_data)
iso_to, iso_back = wals.read_iso_mappings(args.iso_mappings)

for lang in all_languages:
    # TODO Note that gold=pred for train files, proj is what really matters as gold is close to pred.
    sentences = read_sentences("{}/train/{}-ud-train.conllu.lex.sampled_10k.with_{}_pos"
                               .format(path_to_data_files, lang, args.pos_source))

    trigram_freqs_for_sources[lang] = klcpos3.get_trigram_freqs(sentences)

# list of approaches to getting source language distributions for target, granularity, etc.
approaches = {"klcpos3": partial(klcpos3.get_distribution_from_klcpos3,
                                 trigram_fs_of_sources=trigram_freqs_for_sources),
              "langid": partial(get_distributions_from_langid, identifier=identifier_raw),
              "wals": partial(wals.get_distributions_from_wals, identifier=identifier_wals,
                              iso_to=iso_to, iso_back=iso_back, wals=wals_data)}

# read the target language sentences
target_sentences = read_sentences("{}/test/{}-ud-test.conllu.lex.with_{}_pos"
                                  .format(path_to_data_files, target_language, args.pos_source))

# the magic happens here
for approach, get_distribution in approaches.items():
    for granularity in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]:  # sliding window size for estimating source relevance

        # sentence iterator, i = sentence id
        for i in range(granularity, len(target_sentences)+1):

            # the window slides here
            sentence_sample = target_sentences[i-granularity:i]

            # get_distribution() is approach-dependent
            best_source, distribution = get_distribution(target_sentence_sample=sentence_sample,
                                                         list_of_source_languages=source_languages)

            # distribution = softmax(distribution, temperature=0.5)  # TODO Maybe a parameter to play with, but LATER!

            # Assign this best source & distribution to the sample
            for j in range(i - granularity, i):
                if j not in all_mappings[approach][granularity]:  # to avoid reassignment
                    all_mappings[approach][granularity][j] = (best_source, distribution)

    # TODO Add granularity = ALL---is it really necessary? Supposedly the thing converges quickly.

# finally store the results
dill.dump(all_mappings, open("{}/pickles/{}.source_language_mappings.with_{}_pos.pickle"
                             .format(path_to_data_files, target_language, args.pos_source), "wb"))
