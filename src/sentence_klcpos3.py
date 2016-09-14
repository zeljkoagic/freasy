import sys
from collections import defaultdict
import math

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


def get_trigram_freqs(sentences):
    freqs = defaultdict(int)
    sum = 0

    for sentence in sentences:
        c = 0
        while c < len(sentence) - 2:
            freqs[(sentence[c][1], sentence[c + 1][1], sentence[c + 2][1])] += 1
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


#def skl(source_freqs, target_freqs):


ss = read_sentences(sys.argv[1])
st = read_sentences(sys.argv[2])

fs, s = get_trigram_freqs(ss)


for step in range(0, 200, 10):
    ft, _ = get_trigram_freqs(st[:step])
    print(klcpos3(fs, ft, s))


print(len(ss))