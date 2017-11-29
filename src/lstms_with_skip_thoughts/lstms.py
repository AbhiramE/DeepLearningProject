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
from sklearn.preprocessing import MultiLabelBinarizer
import os.path as o

root = o.abspath(o.dirname(__file__))

DATA_FILE = o.join(root, '../../data/booksummaries.txt')
VECTORS_DUMP = o.join(root, '../../data/skip_thought_vectors.p')
FORMATTED_DUMP = o.join(root, '../../data/formatted_data.p')


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
        p.dump(all_data, open(dump, 'wb'))
    else:
        all_data = p.load(open(dump, 'rb'))
    return all_data


def jaccard_similarity(y_true, y_pred):
    y_int = y_true * y_pred
    return -(2 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))


def load_vectors(vectors_file):
    vectors = p.load(open(vectors_file, 'rb'))
    return vectors


def clean_summaries(data_frame):
    new_df = pd.DataFrame(data_frame['summary'].str.split(' ').str.len())
    return data_frame[(new_df['summary'] >= 50) & (new_df['summary'] <= 2500)]


def load_data_and_transform(vector_dump, formatted_dump):
    # Hacks for getting things running on swarm2
    # df = p.load(open(formatted_dump, 'rb'))
    df = read_data(DATA_FILE, use_dump=False)
    df = format_data(df, dump=FORMATTED_DUMP)
    df = clean_summaries(df)
    df = clean_genres(df)
    data_frame = load_vectors(vector_dump)
    alt_frame = df.drop('summary', axis=1)
    print data_frame.shape, alt_frame.shape
    return data_frame, alt_frame


def format_data(data_frame, dump):
    adf = data_frame.drop(['id', 'some_id',
                           'title', 'author', 'rel_date'], axis=1)

    # Drop infrequent genres
    adf = clean_genres(adf[pd.notnull(data_frame['genres'])])
    # One hot encoding the genres.
    mlb = MultiLabelBinarizer()
    adf = adf.join(pd.DataFrame(mlb.fit_transform(adf.pop('genres')),
                                columns=mlb.classes_,
                                index=adf.index))
    p.dump(adf, open(dump, 'wb'))
    return adf


def clean_genres(data_frame):
    alt_frame = data_frame.drop('summary', axis=1)
    data_frame = pd.DataFrame(data_frame['summary'])
    alt_frame = alt_frame[alt_frame.columns[(alt_frame.sum() > 100)]]
    data_frame = pd.concat([data_frame, alt_frame], axis=1)
    return data_frame


def get_train_val_test(vector_dump, formatted_dump):
    X, y = load_data_and_transform(vector_dump=vector_dump, formatted_dump=formatted_dump)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train):
    # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    model = Sequential()
    # model.add(LSTM(100, input_shape=X_train.shape[1:], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    adam = Adam(lr=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    print(model.summary())
    model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=128)
    return model


if __name__ == '__main__':
    # df = read_data(DATA_FILE, use_dump=False)
    # format_df = format_data(df, FORMATTED_DUMP)
    print "Download the skip-thought vectors from Google Drive before you proceed.."
    X_train, X_test, y_train, y_test = get_train_val_test(vector_dump=VECTORS_DUMP, formatted_dump=FORMATTED_DUMP)
    # lstm_model = run_model(X_train, pd.DataFrame.as_matrix(y_train))
