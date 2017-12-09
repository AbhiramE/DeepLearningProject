import cPickle as p
import json
import numpy as np

np.random.seed(1234)

import keras.backend as K
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MultiLabelBinarizer
# from src.explore import data_explore as de
import os.path as o

root = o.abspath(o.dirname(__file__))

DUMP = o.join(root, '../../data/booksummaries.txt')
DATA_DUMP = o.join(root, '../../data/formatted_dump.p')
VEC_DUMP = o.join(root, '../../data/vectors.p')
AVG_DUMP = o.join(root, '../../data/averaged_vectors.p')
MAX_DUMP = o.join(root, '../../data/maxed_out_vectors.p')
LDA_DUMP = o.join(root, '../../data/lda_dump.p')
FC_NET_MODEL = o.join(root, '../../data/fc_net_model.p')


'''
    ###############################################################
    # Gridsearch tried
    # self.num_layers = 3, 4, 5
    # self.layers = [128, 64, 64]
    # self.dropout = 0.3. 0.4, 0.5
    # self.input_dim = 200
    # self.use_batchnorm = True
    # self.learning_rate = 5e-3, 1e-4, 1e-3, 1e-2, 1e-6          
    ###############################################################
'''


def read_data(filename, use_dump, dump=None):
    # Whether or not to use pickle dump
    if not use_dump:
        all_data = pd.read_csv(filename, sep='\t', header=None)
        all_data.columns = ['id', 'some_id', 'title', 'author', 'rel_date', 'genres', 'summary']

        all_data = all_data[pd.notnull(all_data['genres'])]
        genres = pd.Series(all_data['genres']).tolist()
        # Get the first genre in the list of genres
        for i in range(len(genres)):
            genres[i] = [x.encode('utf-8') for x in json.loads(genres[i]).values()]
        all_data['genres'] = pd.Series(genres)
        # p.dump(all_data, open(dump, 'wb'))
    else:
        all_data = p.load(open(dump, 'rb'))
    return all_data


def format_data(data_frame, dump):
    adf = data_frame.drop(['id', 'some_id',
                           'title', 'author', 'rel_date'], axis=1)
    adf = adf[pd.notnull(data_frame['genres'])]

    # One hot encoding the genres.
    mlb = MultiLabelBinarizer()
    adf = adf.join(pd.DataFrame(mlb.fit_transform(adf.pop('genres')),
                                columns=mlb.classes_,
                                index=adf.index))
    p.dump(adf, open(dump, 'wb'))
    return adf


def clean_summaries(data_frame):
    new_df = pd.DataFrame(data_frame['summary'].str.split(' ').str.len())
    return data_frame[(new_df['summary'] >= 50) & (new_df['summary'] <= 2500)]


def clean_genres(data_frame):
    alt_frame = data_frame.drop('summary', axis=1)
    data_frame = pd.DataFrame(data_frame['summary'])
    alt_frame = alt_frame[alt_frame.columns[(alt_frame.sum() > 100)]]
    data_frame = pd.concat([data_frame, alt_frame], axis=1)
    data_frame = data_frame[(data_frame.loc[:, data_frame.columns != 'summary'].T != 0).any()]
    return data_frame


def jaccard_similarity(y_true, y_pred):
    y_int = y_true * y_pred
    return -(2 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))


def load_data_and_transform(vector_dump, data_dump):
    df = read_data(DUMP, use_dump=False)
    df = format_data(df, DATA_DUMP)
    df = clean_summaries(df)
    df = clean_genres(df)
    # df = p.load(open(data_dump, 'rb'))
    data_frame = p.load(open(vector_dump, 'rb'))
    alt_frame = df.drop('summary', axis=1)
    return data_frame, alt_frame


def get_train_val_test(vector_dump, dump):
    X, y = load_data_and_transform(vector_dump, dump)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(2400, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1200))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(600))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    print("[INFO] compiling model...")
    adam = Adam(lr=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    model.fit(X_train, y_train, validation_split=0.2, epochs=60, batch_size=128)
    return model


def predict(model, X_val):
    preds = model.predict(X_val)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    return preds


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, = get_train_val_test(MAX_DUMP, DUMP)
    fc_net_model = run_model(X_train, pd.DataFrame.as_matrix(y_train))
    # results = predict(fc_net_model, X_val)
    # print results
