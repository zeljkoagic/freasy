import sys
import random

random.seed(42)

def read_conll(filename):
	sentences = []
	current = []
	for line in open(filename):
		line = line.strip()
		if line:
			current.append(line)
		else:
			sentences.append(current)
			current = []
	return sentences

sents = read_conll(sys.argv[1])

#if len(sents) > 385:
sents = random.sample(sents, 400)

for sent in sents:
	for token in sent:
		print(token)
	print()
