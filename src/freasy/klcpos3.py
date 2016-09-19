import sys
from collections import defaultdict
import math
from softmax import softmax


def get_trigram_freqs(sentences):
    """
    Provided a list of sentences, each sentence a list of (token, tag) pairs, calculates relative trigram frequencies.
    """
    trigram_fs = defaultdict(int)
    trigram_sum = 0

    for sentence in sentences:
        c = 0
        while c < len(sentence)-2:
            # a token is a tuple (token, tag)
            trigram_fs[(sentence[c][1], sentence[c+1][1], sentence[c+2][1])] += 1
            trigram_sum += 1
            c += 1

    # make the frequencies relative
    for x, y in trigram_fs.items():
        trigram_fs[x] = y / trigram_sum

    return trigram_fs, trigram_sum


def klcpos3(source_trigram_fs, target_trigram_fs, source_sum):
    """
    Calculates the POS trigram-based KL divergence between the source and target distributions.
    """
    kl_score = 0.0

    for trigram, freq in target_trigram_fs.items():
        if trigram not in source_trigram_fs:
            source_trigram_fs[trigram] = 1 / source_sum  # following Rosa & Zabokrtsky (2015)

        kl_score += freq * math.log10(freq / target_trigram_fs[trigram])

    return kl_score


def get_distribution_from_klcpos3(target_sentence_sample, list_of_source_languages, trigram_fs_of_sources):
    """
    Predicts source language appropriateness for a given target sentence sample.
    """
    target_trigram_fs, _ = get_trigram_freqs(target_sentence_sample)

    # for capturing the results
    distribution_of_sources = []
    lowest_kl_score = sys.float_info.max
    best_source = None

    for source_language in list_of_source_languages:
        # calculate KL value for a given sample from the target sentences
        source_trigram_fs, source_trigram_sum = trigram_fs_of_sources[source_language]
        kl_value = klcpos3(source_trigram_fs, target_trigram_fs, source_trigram_sum)
        distribution_of_sources.append((source_language, kl_value))

        # capture the best source language
        if kl_value < lowest_kl_score:
            lowest_kl_score = kl_value
            best_source = source_language

    print(dict(distribution_of_sources))

    return best_source, dict(distribution_of_sources)
