import sys
from collections import defaultdict
import math
from softmax import softmax

def get_trigram_freqs(sentences):
    freqs = defaultdict(int)
    sum = 0

    for sentence in sentences:
        c = 0
        while c < len(sentence) - 2:
            freqs[(sentence[c][1], sentence[c + 1][1], sentence[c + 2][1])] += 1  # token is a tuple, 0=token, 1=tag
            sum += 1
            c += 1

    for x, y in freqs.items():
        freqs[x] = y / sum

    return freqs, sum


def klcpos3(src, tgt, sum):

    kl = 0

    for trigram, freq in tgt.items():
        if trigram not in src:
            # src[trigram] = sys.float_info.min
            src[trigram] = 1 / sum

        # print(trigram, freq, src[trigram])
        kl += freq * math.log10(freq / src[trigram])

    # return math.pow(1/kl, 4)
    return kl


def get_distribution_from_klcpos3(target_sentence_sample, list_of_source_languages, trigram_freqs_of_sources):

    target_trigram_freqs, _ = get_trigram_freqs(target_sentence_sample)

    # for capturing the results
    distribution_of_sources = []
    lowest_kl_score = sys.float_info.max
    best_source = None

    for source_language in list_of_source_languages:
        # calculate KL value for a given sample from the target sentences
        source_trigram_freqs, sum_for_source = trigram_freqs_of_sources[source_language]
        kl_value = klcpos3(source_trigram_freqs, target_trigram_freqs, sum_for_source)
        distribution_of_sources.append((source_language, kl_value))

        # capture the best source language
        if kl_value < lowest_kl_score:
            lowest_kl_score = kl_value
            best_source = source_language

    return best_source, distribution_of_sources
