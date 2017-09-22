import argparse
import dill
from collections import defaultdict
from target_sentence import TargetSentence

# old tacl/udw stuff
# list of languages used in the experiment (WTC constrained), note "ALL" (multi-source delex)
# all_languages = sorted(["ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "he", "hi", "hr", "hu",
#                        "id", "it", "nl", "no", "pl", "pt", "ro", "sl", "sv", "ta", "ALL"])

# added for cleandata repos
all_languages = sorted(["ar", "bg", "ca", "cop", "cs", "cs_cac", "cs_cltt", "cu", "da", "de", "el", "en", "en_esl",
                        "en_lines", "es", "es_ancora", "et", "eu", "fa", "fi", "fi_ftb", "fr", "ga", "gl", "gl_treegal",
                        "got", "grc", "grc_proiel", "he", "hi", "hr", "hu", "id", "it", "ja", "ja_ktc", "kk", "la",
                        "la_ittb", "la_proiel", "lv", "nl", "nl_lassysmall", "no", "pl", "pt", "pt_bosque", "pt_br",
                        "ro", "ru", "ru_syntagrus", "sa", "sk", "sl", "sl_sst", "sv", "sv_lines", "swl", "ta", "tr",
                        "ug", "uk", "vi", "zh"])

# argparse stuff
parser = argparse.ArgumentParser(description="Collect target sentences into a pickle file.")
parser.add_argument("--target_name", required=True, help="target language name")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="root for data files")
args = parser.parse_args()

assert args.target_name in all_languages, "Unknown language: {}".format(args.target_name)

# old tacl/udw stuff
# add the target first; note this is a *gold* POS file
# handles = [open("{}/test/{}-ud-test.conllu.lex.with_gold_pos".format(args.data_root, args.target_name))]

# added for cleandata repos
handles = [open("{}/ud14/train/{}-ud-train.no_sent_id.no_mwe.no_feat.basic_deprel.ceil_10k.{}_pos.delex.conll_2009"
                .format(args.data_root, args.target_name, args.pos_source))]

source_languages = sorted(set(all_languages) - {args.target_name})
source_languages_per_handle = []  # remember the sequence of source languages, just in case

# open all the source file handles
for source_language in source_languages:
    assert source_language != args.target_name, "Target cannot be source!"
    # old tacl/udw stuff
    #handles.append(open("{}/test/{}-ud-test.conllu.delex.sampled_10k.with_{}_pos.parsed_with_{}"
    #                    .format(args.data_root, args.target_name, args.pos_source, source_language)))
    # added for cleandata repos
    handles.append(open("{}/ud14/train/{}-ud-train.no_sent_id.no_mwe.no_feat.basic_deprel.ceil_10k.{}_pos.delex.conll_2009.parsed_by_{}"
                       .format(args.data_root, args.target_name, args.pos_source, source_language)))
    source_languages_per_handle.append(source_language)  # record the sequence

# list of target sentences as final result of the script
target_sentences = []

# buffers
current_tokens = []
current_gold_pos = []
current_predicted_pos = []
current_gold_heads = []
current_multi_source_heads = []
current_single_source_heads = defaultdict(list)  # indexed by source language name
current_token_id = 0
current_target_sentence_id = 0

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
        current_gold_pos.append(pos)
        current_gold_heads.append(int(head))

        # and then get all the source stuff
        for idx, source_line in enumerate(source_lines):
            source_language = source_languages_per_handle[idx]

            _, _, _, _, ppos, _, _, _, _, phead, _, pdeprel, _, _ = source_line.split("\t")

            if idx == 0:
                current_predicted_pos.append(ppos)

            # distinction between single- and multi-source parsers
            if source_language == "ALL":
                current_multi_source_heads.append(int(phead))
            else:
                current_single_source_heads[source_language].append(int(phead))

    else:

        # todo create sentence tensors
        # todo we use tensors instead of arc lists from now on, act accordingly

        # create and append the sentence
        current_sentence = TargetSentence(idx=current_target_sentence_id,
                                          language=args.target_name,
                                          tokens=current_tokens,
                                          gold_heads=current_gold_heads,
                                          gold_pos=current_gold_pos,
                                          predicted_pos=current_predicted_pos,
                                          multi_source_heads=current_multi_source_heads,
                                          single_source_heads=current_single_source_heads,
                                          pos_source_type=args.pos_source)

        target_sentences.append(current_sentence)

        # clear the buffers
        current_tokens = []
        current_gold_pos = []
        current_predicted_pos = []
        current_gold_heads = []
        current_multi_source_heads = []
        current_single_source_heads = defaultdict(list)
        current_target_sentence_id += 1  # this one provides sentence ids

# finally store the pickle
dill.dump(target_sentences, open("{}/pickles/{}.as_target_language.all_parses.TRAIN_DATA.with_{}_pos.pickle"
                                 .format(args.data_root, args.target_name, args.pos_source), "wb"))
