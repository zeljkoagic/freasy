import argparse
import dill
from collections import defaultdict
from target_sentence import Arc, TargetSentence

# list of languages used in the experiment (WTC constrained)
langs = sorted(["ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "he", "hi", "hr", "hu", "id",
                "it", "nl", "no", "pl", "pt", "ro", "sl", "sv", "ta", "ALL"])

# POS tag sources in the experiment: gold, predicted (in-language), projected-predicted (cross-language)
pos_sources = ["gold", "pred", "proj"]

# argparse stuff
parser = argparse.ArgumentParser(description="Collect target sentences into a pickle file.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
args = parser.parse_args()

target_lang = args.target_name
path_to_data_files = args.data_root

assert target_lang in langs, "Unknown language: {}".format(target_lang)

# add the target first
# TODO Don't forget that this is a *gold POS* file
handles = [open("{}/test/{}-ud-test.conllu.lex.with_gold_pos".format(path_to_data_files, target_lang))]

# this remembers the sequence of POS sources and source languages, just in case
pos_sources_per_handle = []
source_langs_per_handle = []

source_languages = sorted(set(langs) - {target_lang})

# open all the source file handles x different POS sources
for source_lang in source_languages:
    for pos_source in pos_sources:
        assert source_lang != target_lang, "Target cannot be source!"
        handles.append(open("{}/test/{}-ud-test.conllu.delex.sampled_10k.with_{}_pos."
                            "parsed_with_{}".format(path_to_data_files, target_lang, pos_source, source_lang)))

        # record the sequence
        pos_sources_per_handle.append(pos_source)
        source_langs_per_handle.append(source_lang)

# list of target sentences as final result of the script
target_sentences = []

# buffers
current_tokens = []
current_pos = defaultdict(list)  # indexed by POS source
current_gold_arcs = []
current_source_arcs = defaultdict(lambda: defaultdict(list))  # indexed by language and POS source
current_token_id = 0
current_target_sentence_id = 0  # important, stores sentence ids

# iterate through *all* the file handles line by line
for lines in zip(*handles):

    # separate the target line from the source lines for special treatment
    target_line = lines[0].strip()
    source_lines = lines[1:]

    if target_line:
        current_token_id += 1
        # get target data from target line
        _, token, _, _, pos, _, _, _, head, _, deprel, _, _ = target_line.split("\t")

        # add all the gold stuff
        current_tokens.append(token)
        current_pos["gold"].append(pos)
        current_gold_arcs.append(Arc(lang=target_lang, head=int(head), dependent=current_token_id, deprel=deprel,
                                     pos_source="gold", weight=1.0))

        # and then get all the source stuff
        for idx, source_line in enumerate(source_lines):
            source_lang = source_langs_per_handle[idx]
            pos_source = pos_sources_per_handle[idx]

            _, _, _, _, pos, _, _, _, _, phead, _, pdeprel, _, _ = source_line.split("\t")

            # harvest the *other* POS tags (i.e., non-gold), and do it only once as all parses have the same
            if pos_source != "gold" and (idx == 1 or idx == 2):
                current_pos[pos_source].append(pos)

            # add an arc to the list
            current_source_arcs[source_lang][pos_source].append(Arc(lang=source_lang, head=int(phead),
                                                                    dependent=current_token_id, deprel=pdeprel,
                                                                    pos_source=pos_source, weight=1.0))

    else:
        # create and append the sentence
        current_sentence = TargetSentence(idx=current_target_sentence_id,
                                          lang=target_lang,
                                          tokens=current_tokens,
                                          gold_arcs=current_gold_arcs,
                                          pos=current_pos,
                                          arcs_from_sources=current_source_arcs)

        target_sentences.append(current_sentence)

        # clear the buffers
        current_tokens = []
        current_gold_arcs = []
        current_pos = defaultdict(list)
        current_source_arcs = defaultdict(lambda: defaultdict(list))
        current_token_id = 0
        current_target_sentence_id += 1  # this one provides sentence ids

# finally store the pickle
dill.dump(target_sentences, open("{}/pickles/{}.as_target_language.all_parses.pickle".format(path_to_data_files,
                                                                                             target_lang), "wb"))
