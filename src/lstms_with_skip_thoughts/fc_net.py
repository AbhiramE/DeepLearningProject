import cPickle as p

import matplotlib.pyplot as plt
from LogCallbacks import Logger
import numpy as np
import keras.backend as K
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Dropout, Activation, PReLU
from keras import regularizers
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
import os.path as o

root = o.abspath(o.dirname(__file__))

np.random.seed(1234)

DUMP = o.join(root, '../../data/formatted_data.p')
AVG_DUMP = o.join(root, '../../data/averaged_vectors.p')
MAX_DUMP = o.join(root, '../../data/maxed_out_vectors.p')
FC_NET_MODEL = o.join(root, '../../data/fc_net_model.p')


def jaccard_similarity(y_true, y_pred):
    y_int = y_true * y_pred
    return K.logsumexp(-(2.0 * K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred))))


def load_data_and_transform(lda_dump, data_dump):
    df = p.load(open(data_dump, 'rb'))
    data_frame = p.load(open(lda_dump, 'rb'))
    return data_frame, df


def get_train_val_test(lda_dump, dump):
    X, y = load_data_and_transform(lda_dump=lda_dump, data_dump=dump)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def run_model(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    model = Sequential()

    # Layer 1
    model.add(Dense(2400, input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=regularizers.l2(1e-4), alpha_constraint=None))
    model.add(Dropout(0.3))

    # Layer 2
    model.add(Dense(1200, kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)))

    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros',
                    alpha_regularizer=regularizers.l2(1e-4),
                    alpha_constraint=None))
    model.add(Dropout(0.3))

    # Layer 3
    model.add(Dense(600,
                    kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)
                    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=regularizers.l2(1e-4), alpha_constraint=None))
    model.add(Dropout(0.3))

    # Layer 4
    model.add(Dense(300,
                    kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)
                    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=regularizers.l2(1e-4), alpha_constraint=None))
    model.add(Dropout(0.3))

    # Layer 5
    model.add(Dense(150,
                    kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)
                    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros',
                    alpha_regularizer=regularizers.l2(1e-4),
                    alpha_constraint=None))
    model.add(Dropout(0.3))

    # Layer 6
    model.add(Dense(75,
                    kernel_regularizer=regularizers.l2(1e-3),
                    bias_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l2(0)
                    ))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer='zeros', alpha_regularizer=regularizers.l2(1e-4), alpha_constraint=None))
    model.add(Dropout(0.3))

    # Output
    model.add(Dense(y_train.shape[1], activation='sigmoid'))

    logger = Logger(validation_data=(X_val, y_val))

    print("[INFO] compiling model...")
    adam = Adam(lr=5e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[jaccard_similarity])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=400, batch_size=128, callbacks=[logger])

    return model, logger, history


def plot_loss(logger):
    train_loss = logger.train_loss
    x_axis = range(len(train_loss))
    sns_plot = sns.tsplot(data=train_loss, time=x_axis, value='Loss', legend=True)
    plt.show()
    fig = sns_plot.get_figure()
    fig.savefig('../../data/loss.png')


def plot_losses(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig = plt.figure()
    x_axis = range(len(train_loss))
    plt.plot(x_axis, train_loss, 'b-', label='Training Loss')
    plt.plot(x_axis, val_loss, 'g-', label='Validation Loss')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('../../data/validation_vs_training_loss.png')


def plot_metrics(logger):
    jaccard = pd.Series(logger.jaccard_similarity)
    metric1 = pd.Series(logger.metric1_array)
    metric2 = pd.Series(logger.metric2_array)

    fig = plt.figure()
    plt.plot(range(len(jaccard)), jaccard, 'b-', label='Jaccard')
    plt.plot(range(len(metric1)), metric1, 'g-', label='Best 1 metric')
    plt.plot(range(len(metric2)), metric2, 'r-', label='Best k metric')
    plt.legend(loc='best')
    plt.show()
    fig.savefig('../../data/metrics.png')


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


def test(model, X_val, df):
    y_pred = model.predict(X_val)
    sample = np.random.randint(0, 20)
    pred_indices = np.argsort(y_pred[sample])[-3:][::-1]

    print "\n \n \nLets try a test sample..."
    print(df.as_matrix()[sample][0])
    print (df.columns[pred_indices])


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, = get_train_val_test(AVG_DUMP, DUMP)
    print X_train.shape
    fc_net_model, logger, hist = run_model(X_train, pd.DataFrame.as_matrix(y_train.drop('summary', axis=1)))
    plot_loss(logger)
    plot_losses(hist)
    plot_metrics(logger)
    # results = predict(fc_net_model, X_test, y_test.as_matrix())
    # print(results)
    # print results
