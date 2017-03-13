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

# 1. map data to one-hot
# 2. create arch
# 3. train

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
    lang, idx, poss, ranks = item
    all = []
    for pos in poss:
        all.append(float(one_hot[pos]) / float(len(one_hot)))

    all = np.array(all, dtype=float)

    X_train.append(all)
    ranks[lang] = 0
    ranks = softmax(ranks)
    yval = np.array([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False) if y in dev_langs], dtype=float)
    am = np.argmax(yval)
    yval2 = [float(i == am) for i, _ in enumerate(yval)]
    Y_train.append(yval2)

for item in test_data:
    lang, idx, poss, ranks = item
    all = []
    for pos in poss:
        all.append(float(one_hot[pos]) / float(len(one_hot)))

    all = np.array(all, dtype=float)

    X_test.append(all)
    ranks[lang] = 0
    ranks = softmax(ranks)
    yval = np.array([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False) if y in dev_langs], dtype=float)
    am = np.argmax(yval)
    yval2 = [float(i == am) for i, _ in enumerate(yval)]  # if categorical, and not softmax
    Y_test.append(yval2)


X_train = sequence.pad_sequences(X_train, maxlen=128, dtype=float)
X_test = sequence.pad_sequences(X_test, maxlen=128, dtype=float)

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 128, 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 128, 1))

print(X_test_reshaped[0], Y_test[0])

model = Sequential()

model.add(LSTM(output_dim=256,
               input_dim=1,
               input_length=128,
               activation="relu",
               return_sequences=True))

model.add(LSTM(output_dim=128,
               input_dim=256,
               input_length=128,
               activation="relu",
               return_sequences=True))

model.add(LSTM(output_dim=64,
               input_dim=256,
               input_length=128,
               activation="relu",
               return_sequences=True))

model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train_reshaped, Y_train,
          batch_size=32,
          nb_epoch=5,
          validation_data=[X_test_reshaped, Y_test])

model.evaluate(X_test_reshaped, Y_test)
