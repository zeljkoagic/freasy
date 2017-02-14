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

one_hot = {
    "ADJ":   0,
    "ADP":   1,
    "ADV":   2,
    "AUX":   3,
    "CONJ":  4,
    "DET":   5,
    "INTJ":  6,
    "NOUN":  7,
    "NUM":   8,
    "PART":  9,
    "PRON":  10,
    "PROPN": 11,
    "PUNCT": 12,
    "SCONJ": 13,
    "SYM":   14,
    "VERB":  15,
    "X":     16
}

X_train = []
Y_train = []

for item in training_data:
    lang, idx, poss, ranks = item
    all = []
    for pos in poss:
        all.append(one_hot[pos])

    all = np.array(all, dtype=float)
    all /= 17.0

    X_train.append(all)
    ranks[lang] = 0
    ranks = softmax(ranks)
    yval = np.array([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False)], dtype=float)
    am = np.argmax(yval)
    yval2 = [float(i == am) for i, _ in enumerate(yval)]
    Y_train.append(yval2) # if y in dev_langs])
    print(yval2)

X_train = np.array(X_train[:-100])
Y_train = np.array(Y_train[:-100])

X_test = np.array(X_train[-100:])
Y_test = np.array(Y_train[-100:])

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

print(X_train[0], Y_train[0])

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=10)
X_test = sequence.pad_sequences(X_test, maxlen=10)
print('X_train shape:', X_train.shape)

#X_train_reshaped = np.reshape(X_train, (3228, 100, 1))
#print('X_train reshaped:', X_train_reshaped.shape)
#print(X_train_reshaped)

model = Sequential()

#input_array = np.random.randint(17, size=(333, 10))  # 333 datapoints, 10 items each, values from 0 to 16
#output_array = np.random.randint(8, size=(333, 3))  # 333 datapoints, 3 items each, values from 0 to 8

model.add(Embedding(input_dim=17,
                    output_dim=128,
                    input_length=10,
                    mask_zero=False))

#model.add(Dense(output_dim=64,
#                input_dim=128, activation="relu"))

model.add(LSTM(output_dim=64,
               input_dim=128,
               input_length=10,
               activation="relu",
               return_sequences=True))

model.add(LSTM(output_dim=32,
               input_dim=64,
               input_length=10,
               activation="relu",
               return_sequences=False))

model.add(Dense(26, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


#output_array = model.predict(X_train)
#print(X_train.shape, output_array.shape)


print('Train...')
model.fit(X_train, Y_train,
          batch_size=32,
          nb_epoch=10,
          validation_data=[X_test, Y_test])
