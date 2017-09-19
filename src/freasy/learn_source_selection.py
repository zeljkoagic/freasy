# here we use a recurrent neural network to learn the target->source mapping

import argparse
import dill
import operator
from softmax import softmax
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU


def to_ranks(lasvals):
    """ Converts LAS values to source ranks. WHAT ABOUT TIES???
    """
    array = np.array(lasvals)
    temp = array.argsort()
    tranks = np.empty(len(array), int)
    tranks[temp] = np.arange(len(array))
    return tranks+1


parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
args = parser.parse_args()

dev_langs = ["en", "es", "de", "fr", "it", "hi", "hr", "cs", "he", "id"]
test_langs = ["pt", "pl", "da"]

training_data = []
test_data = []

for lang in dev_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    training_data += target_sentences

for lang in test_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    test_data += target_sentences

one_hot = {
    "ADJ":   1,
    "ADP":   2,
    "ADV":   3,
    "AUX":   4,
    "CONJ":  5,
    "DET":   6,
    "INTJ":  7,
    "NOUN":  8,
    "NUM":   9,
    "PART":  10,
    "PRON":  11,
    "PROPN": 12,
    "PUNCT": 13,
    "SCONJ": 14,
    "SYM":   15,
    "VERB":  16,
    "X":     17
}

X_train = []
Y_train = []
X_test = []
Y_test = []

for item in training_data:
    lang, idx, poss, ranks = item  # get the training instance: target lang, sentence id, list of POS tags, corr heads

    # filter out the sentences that are too short or too long
    n_tokens = len(poss)
    if n_tokens < 20 or n_tokens > 50:
        continue

    # translate the POS tags into floats
    all = []
    for pos in poss:
        all.append(float(one_hot[pos]))
    all = np.array(all, dtype=float)

    # add training instance
    X_train.append(all)

    ranks[lang] = 0  # target language does not participate
    ranks2 = {langg: ranks[langg] for langg in dev_langs}
    ranks2 = softmax(ranks2, temperature=0.1)  # softmax the correct head counts
    y_val = np.array([x for y, x in sorted(ranks2.items(), key=operator.itemgetter(0), reverse=False) if y in dev_langs], dtype=float)
    am = np.argmax(y_val)
    y_val = np.zeros_like(y_val)
    y_val[am] = 1
    Y_train.append(y_val.tolist())

for item in test_data:
    lang, idx, poss, ranks = item  # get the training instance: target lang, sentence id, list of POS tags, corr heads

    # filter out the sentences that are too short or too long
    n_tokens = len(poss)
    if n_tokens < 20 or n_tokens > 50:
        continue

    # translate the POS tags into floats
    all = []
    for pos in poss:
        all.append(float(one_hot[pos]))
    all = np.array(all, dtype=float)

    # add training instance
    X_test.append(all)

    ranks[lang] = 0  # target language does not participate
    ranks2 = {langg:ranks[langg] for langg in dev_langs}
    ranks2 = softmax(ranks2, temperature=0.1)  # softmax the correct head counts
    y_val = np.array([x for y, x in sorted(ranks2.items(), key=operator.itemgetter(0), reverse=False) if y in dev_langs], dtype=float)
    am = np.argmax(y_val)
    y_val = np.zeros_like(y_val)
    y_val[am] = 1
    Y_test.append(y_val.tolist())


X_train = sequence.pad_sequences(X_train, maxlen=64, dtype=float)
X_test = sequence.pad_sequences(X_test, maxlen=64, dtype=float)

#X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 64, 1))
#X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 64, 1))

#print(X_test_reshaped[0], Y_test[0])

model = Sequential()

model.add(Embedding(len(one_hot)+1, 10))

model.add(LSTM(output_dim=32,
               input_dim=10,
               input_length=64,
               activation="sigmoid",
               return_sequences=False))

model.add(Dense(10, activation='softmax'))

model.compile('adam', 'mse', metrics=['accuracy'])

print('Train...')
model.fit(X_train, Y_train,
          batch_size=32,
          nb_epoch=100,
          validation_data=[X_test, Y_test])

