# here we use a recurrent neural network to learn the target->source mapping

import argparse
import dill
import operator
from softmax import softmax
import numpy as np
import tensorflow as tf

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, GRU

parser = argparse.ArgumentParser(description="TODO")
parser.add_argument("--data_root", required=True, help="root for data files")
parser.add_argument("--pos_source", required=True, choices=["gold", "pred", "proj"], help="POS source")
args = parser.parse_args()

dev_langs = ["en", "de", "es"]

training_data = []

for lang in dev_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    training_data += target_sentences

# 1. map data to one-hot
# 2. create arch
# 3. train

one_hot = {
    "ADJ":   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ADP":   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ADV":   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "AUX":   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "CONJ":  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "DET":   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "INTJ":  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NOUN":  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "NUM":   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "PART":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "PRON":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "PROPN": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "PUNCT": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "SCONJ": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "SYM":   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "VERB":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "X":     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

#one_hot = {
#    "ADJ":   1,
#    "ADP":   2,
#    "ADV":   3,
#    "AUX":   4,
#    "CONJ":  5,
#    "DET":   6,
#    "INTJ":  7,
#    "NOUN":  8,
#    "NUM":   9,
#    "PART":  10,
#    "PRON":  11,
#    "PROPN": 12,
#    "PUNCT": 13,
#    "SCONJ": 14,
#    "SYM":   15,
#    "VERB":  16,
#    "X":     17
#}


X_train = []
Y_train = []

for item in training_data:
    lang, idx, poss, ranks = item
    all = []
    for pos in poss:
        all += one_hot[pos]
    X_train.append(all)
    ranks[lang] = 0
    ranks = softmax(ranks)
    Y_train.append([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False)]) # if y in dev_langs])

X_train = np.array(X_train[:-100])
Y_train = np.array(Y_train[:-100])

X_test = np.array(X_train[-100:])
Y_test = np.array(Y_train[-100:])

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

print(X_train[0], Y_train[0])

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=1000)
X_test = sequence.pad_sequences(X_test, maxlen=1000)
print('X_train shape:', X_train.shape)

X_train_reshaped = np.reshape(X_train, (3228, 100, 10))
#print('X_train reshaped:', X_train_reshaped.shape)

model = Sequential()

# model.add(Embedding(40000, 512))

model.add(LSTM(128, activation="sigmoid", input_dim=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(256, activation="sigmoid", return_sequences=False)))
model.add(Dropout(0.2))

model.add(Dense(26, activation='softmax'))

model.compile('adam', 'kullback_leibler_divergence', metrics=['accuracy'])

print('Train...')
model.fit(X_train_reshaped, Y_train,
          batch_size=32,
          nb_epoch=100,
          validation_data=[X_test, Y_test])
