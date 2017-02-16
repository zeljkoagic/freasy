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

dev_langs = ["en", "es", "de", "fr", "it", "hi", "hr", "cs", "he", "id"]

training_data = []

for lang in dev_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    training_data += target_sentences

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

for item in training_data:
    lang, idx, poss, ranks = item
    all = []
    for pos in poss:
        all.append(one_hot[pos])

    all = np.array(all, dtype=float)
   #all /= 17.0

    X_train.append(all)
    ranks[lang] = 0
    ranks = softmax(ranks)
    yval = np.array([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False) if y in dev_langs], dtype=float)
    am = np.argmax(yval)
    yval2 = [float(i == am) for i, _ in enumerate(yval)]
    Y_train.append(yval2)
    #print(yval2)

X_train = np.array(X_train[:-100])
Y_train = np.array(Y_train[:-100])

X_test = np.array(X_train[-100:])
Y_test = np.array(Y_train[-100:])


print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=30)
X_test = sequence.pad_sequences(X_test, maxlen=30)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 30, 1))
X_test_reshaped = np.reshape(X_test, (100, 30, 1))

model = Sequential()

#model.add(Embedding(input_dim=18,
#                    output_dim=128,
#                    input_length=10))


model.add(LSTM(output_dim=64,
                             input_dim=1,
                             input_length=30,
                             activation="relu",
                             return_sequences=True,
                             dropout_U=0.2,
                             dropout_W=0.2))

model.add(LSTM(output_dim=32,
                             # input_dim=64,
                             # input_length=10,
                             activation="relu",
                             return_sequences=False,
                             dropout_U=0.2,
                             dropout_W=0.2))

model.add(Dense(10, activation='softmax'))

model.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train_reshaped, Y_train,
          batch_size=32,
          nb_epoch=500,
          validation_data=[X_test_reshaped, Y_test])
