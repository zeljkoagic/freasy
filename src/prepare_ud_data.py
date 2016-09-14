import sys

delex = int(sys.argv[2])

for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        line = line.split("\t")
        if len(line) == 10:
            idx, token, lemma, cpos, fpos, feats, head, deprel, _, _ = line

            if "-" in idx:
                continue

            if delex != 0:
                token = "_"

            # print("%s\t_\t_\t%s\t_\t_\t%s\t%s\t_\t_" % (idx, cpos, head, deprel))  # conll 2006
            print("%s\t%s\t_\t_\t%s\t_\t_\t_\t%s\t_\t%s\t_\t_" % (idx, token, cpos, head, deprel))  # conll 2009
    else:
        print()
