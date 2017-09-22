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

train_langs = ["en", "es", "de", "fr", "it", "hi", "hr", "cs", "he", "id"]
test_langs = ["pt", "pl", "da"]

training_data = []
test_data = []

for lang in train_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    training_data += target_sentences

for lang in test_langs:
    target_sentences = dill.load(open("{}/pickles/target_lang_{}.pos_source_{}.nn_training_data"
                                      .format(args.data_root, lang, args.pos_source), "rb"))
    test_data += target_sentences

ud_pos_tags = sorted(["ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
                      "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"])

tag_ids = {x: i for i, x in enumerate(ud_pos_tags)}  # tag mapping to ids

X_train = []
Y_train = []
X_test = []
Y_test = []

for data in [(test_data, X_test, Y_test), (training_data, X_train, Y_train)]:
    the_data, X, Y = data
    for item in the_data:
        # get the training instance: target lang, sentence id, list of POS tags, corr heads
        target_lang, idx, poss, ranks = item

        # filter out the sentences that are too short or too long
        n_tokens = len(poss)
        #if n_tokens < 20 or n_tokens > 50:
        #    continue
        if n_tokens > 5:
            continue

        # translate the POS tags into floats, and add training instance
        encoded_pos_sequence = np.array([tag_ids[pos] for pos in poss], dtype=float)
        X.append(encoded_pos_sequence)

        # convert reverse ranks to softmax
        ranks[target_lang] = 0
        ranks = softmax({lang: ranks[lang] for lang in train_langs}, temperature=0.1)
        # sort by key to avoid python dictionary shite
        y_val = np.array([x for y, x in sorted(ranks.items(), key=operator.itemgetter(0), reverse=False)
                          if y in train_langs], dtype=float)
        # put all to 0 save for the single best source
        index_of_max = np.argmax(y_val)
        y_val = np.zeros_like(y_val)
        y_val[index_of_max] = 1
        # add to training data
        Y.append(y_val.tolist())

X_train = sequence.pad_sequences(X_train, maxlen=64, dtype=float)
X_test = sequence.pad_sequences(X_test, maxlen=64, dtype=float)

model = Sequential()

model.add(Embedding(len(tag_ids)+1, 12))

model.add(LSTM(units=64,
               input_shape=(64, 12),
               activation="relu",
               return_sequences=False,
               dropout=0.2,
               recurrent_dropout=0.2))

model.add(Dense(10, activation='softmax'))

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(X_train, Y_train,
          batch_size=32,
          epochs=100,
          validation_data=[X_test, Y_test])

