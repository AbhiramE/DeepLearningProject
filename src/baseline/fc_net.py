import cPickle as p

import keras.backend as K
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization

DUMP = '../../data/formatted_data.p'
LDA_DUMP = '../../data/lda_dump.p'
FC_NET_MODEL = '../../data/fc_net_model.p'


def jaccard_similarity(y_true, y_pred):
        y_int = y_true * y_pred
        return -(2 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))


def load_data_and_transform(lda_dump, data_dump):
    df = p.load(open(data_dump, 'rb'))
    data_frame = p.load(open(lda_dump, 'rb'))
    alt_frame = df.drop('summary', axis=1)
    return data_frame, alt_frame


def get_train_val_test(lda_dump, dump):
    X, y = load_data_and_transform(lda_dump=lda_dump, data_dump=dump)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train):
    model = Sequential()
    model.add(Dense(256, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    print("[INFO] compiling model...")
    adam = Adam(lr=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    model.fit(X_train, y_train, validation_split=0.2, epochs=60, batch_size=128)
    p.dump(model, open(FC_NET_MODEL, 'wb'))
    return model


def predict(model, X_val):
    preds = model.predict(X_val)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    return preds


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, = get_train_val_test(LDA_DUMP, DUMP)
    print X_train.shape
    fc_net_model = run_model(X_train, pd.DataFrame.as_matrix(y_train))
    # results = predict(fc_net_model, X_val)
    # print results
