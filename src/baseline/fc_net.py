import cPickle as p

from LogCallbacks import Logger 
import numpy as np
import keras.backend as K
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import os.path as o

root = o.abspath(o.dirname(__file__))

np.random.seed(1234)

DUMP = o.join(root, '../../data/formatted_data.p')
LDA_DUMP = o.join(root, '../../data/lda_dump.p')
FC_NET_MODEL = o.join(root, '../../data/fc_net_model.p')


def jaccard_similarity(y_true, y_pred):
    y_int = y_true * y_pred
    return -(2 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))


def mutli_accuracy(y_true, y_pred):
    print type(K.eval(y_pred))
    '''
    y_true_cols = np.count_nonzero(y_true, axis=1)
    correct_pred = 0
    for i in xrange(len(y_pred)):
        pred_indices = np.argsort(y_pred[i])[-y_true_cols[i]:][::-1]
        true_indices = np.argsort(y_true[i])[-y_true_cols[i]:][::-1]
        if len(np.intersect1d(pred_indices, true_indices)) > 0:
            correct_pred += 1

    return K.tf.convert_to_tensor(float(correct_pred) / len(y_pred), np.float32)
    '''
    return y_pred


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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    v_data = (X_val, y_val)
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

    logger = Logger(v_data) 

    print("[INFO] compiling model...")
    adam = Adam(lr=5e-3)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    model.fit(X_train, y_train, validation_data = v_data , epochs=35,batch_size=128, callbacks= [logger])
    return model


def predict(model, X_val, y_true):
    y_pred = model.predict(X_val)
    y_true_cols = np.count_nonzero(y_true, axis=1)
    correct_pred = 0
    for i in xrange(len(y_pred)):
        pred_indices = np.argsort(y_pred[i])[-y_true_cols[i]:][::-1]
        true_indices = np.argsort(y_true[i])[-y_true_cols[i]:][::-1]
        if len(np.intersect1d(pred_indices, true_indices)) > 0:
            correct_pred += 1

    return float(correct_pred) / len(y_pred)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, = get_train_val_test(LDA_DUMP, DUMP)
    print X_train.shape
    fc_net_model = run_model(X_train, pd.DataFrame.as_matrix(y_train))
    results = predict(fc_net_model, X_test, y_test.as_matrix())
    print(results)
    # print results
