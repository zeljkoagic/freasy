import sys

gold = open(sys.argv[1]).readlines()
system = open(sys.argv[2]).readlines()

for g, s in zip(gold, system):
    g = g.strip()
    s = s.strip()
    if g and s and len(g.split()) > 1:
        gl = g.split()
        sl = s.split()

	print("%s\t%s\t_\t_\t%s\t_\t_\t_\t%s\t_\t%s\t_\t_" % (gl[0], gl[1], sl[5], gl[8], gl[10]))  # conll 2009 lex
	# print("%s\t_\t_\t_\t%s\t_\t_\t_\t%s\t_\t%s\t_\t_" % (gl[0], sl[5], gl[8], gl[10]))  # conll 2009 delex
    else:
        print
