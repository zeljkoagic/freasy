import sys

gold = open(sys.argv[1]).readlines()
system = open(sys.argv[2]).readlines()

correct = 0.0
total = 0.0

for g, s in zip(gold, system):
    g = g.strip()
    s = s.strip()
    if g and s and len(g.split()) > 1:
        gl = g.split()[10]
        sl = s.split()[11]
	gh = g.split()[8]
        sh = s.split()[9]
        if gh == sh:
            correct += 1.0
        total += 1.0

print(correct / total)
