import cPickle as p
import keras.backend as K
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization

DATA_FILE = '../../data/booksummaries.txt'
VECTORS_DUMP = '../../data/skip_thought_vectors.p'
FORMATTED_DUMP = '../../data/formatted_data.p'


def jaccard_similarity(y_true, y_pred):
    y_int = y_true * y_pred
    return -(2 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))


def load_vectors(vectors_file):
    vectors = p.load(open(vectors_file, 'rb'))
    return vectors


def load_data_and_transform(vector_dump, formatted_dump):
    df = p.load(open(formatted_dump, 'rb'))
    data_frame = load_vectors(vector_dump)
    alt_frame = df.drop('summary', axis=1)
    print data_frame.shape, alt_frame.shape
    return data_frame, alt_frame


def get_train_val_test(vector_dump, formatted_dump):
    X, y = load_data_and_transform(vector_dump=vector_dump, formatted_dump=formatted_dump)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train):
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(100, input_shape=X_train.shape[1:], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    adam = Adam(lr=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    print(model.summary())
    model.fit(X_train, y_train, validation_split=0.2, epochs=3, batch_size=64)
    return model


if __name__ == '__main__':
    print "Download the skip-thought vectors from Google Drive before you proceed.."
    X_train, X_test, y_train, y_test = get_train_val_test(vector_dump=VECTORS_DUMP, formatted_dump=FORMATTED_DUMP)
    lstm_model = run_model(X_train, pd.DataFrame.as_matrix(y_train))
