import sys
import pickle
from collections import defaultdict
from target_sentence import Arc, TargetSentence
import dill

langs = sorted(["ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fa", "fi", "fr", "he", "hi", "hr", "hu", "id",
                "it", "nl", "no", "pl", "pt", "ro", "sl", "sv", "ta", "ALL"])

pos_sources = ["gold", "pred", "proj"]

target_lang = sys.argv[1]
path_to_files = sys.argv[2]
assert target_lang in langs, "Unknown language: {}".format(target_lang)

handles = [open("{}/{}-ud-test.conllu.lex".format(path_to_files, target_lang))]
pos_sources_per_handle = []
source_langs_per_handle = []

for source_lang in set(langs) - set([target_lang]):
    for pos_source in pos_sources:
        assert source_lang != target_lang, "Target cannot be source!"
        handles.append(open("{}/{}-ud-test.conllu.delex.sampled_10k.with_{}_pos."
                            "parsed_with_{}".format(path_to_files, target_lang, pos_source, source_lang)))

        pos_sources_per_handle.append(pos_source)
        source_langs_per_handle.append(source_lang)

target_sentences = []
current_tokens = []
current_pos = defaultdict(list)
current_gold_arcs = []
current_source_arcs = defaultdict(lambda: defaultdict(list))
current_token_id = 0
current_target_sentence_id = 0

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
        current_gold_arcs.append(Arc(lang=target_lang, source=int(head), target=current_token_id, deprel=deprel,
                                     pos_source="gold", weight=1.0))

        # and then get all the source stuff
        for idx, source_line in enumerate(source_lines):
            source_lang = source_langs_per_handle[idx]
            pos_source = pos_sources_per_handle[idx]

            _, _, _, _, pos, _, _, _, _, phead, _, pdeprel, _, _ = source_line.split("\t")

            # harvest the POS
            if pos_source != "gold" and (idx == 1 or idx == 2):
                current_pos[pos_source].append(pos)

            # add an arc to the list
            current_source_arcs[source_lang][pos_source].append(Arc(lang=source_lang, source=int(phead),
                                                                    target=current_token_id, deprel=pdeprel,
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
        current_target_sentence_id += 1

dill.dump(target_sentences, open("{}.as_target_language.all_parses.pickle".format(target_lang), "wb"))

#pickle.dump(target_sentences, open("{}.as_target_language.all_parses.pickle".format(target_lang), "wb"))

#corr_pred = 0
#corr_proj = 0
#total = 0
#for s in target_sentences:
#    corr_pred += sum([pred_pos == gold_pos for pred_pos, gold_pos in zip(s.pos["pred"], s.pos["gold"])])
#    corr_proj += sum([proj_pos == gold_pos for proj_pos, gold_pos in zip(s.pos["proj"], s.pos["gold"])])
#    total += len(s.tokens)

#print(corr_pred/total, corr_proj/total)

#print(target_sentences[1].pos["gold"])
#print(target_sentences[1].pos["proj"])