from keras import backend as K
from sklearn.metrics import jaccard_similarity_score
import numpy as np
from keras.callbacks import Callback

CLASS_THRESHOLD = [0.3, 0.3, 0.2, 0.3, 0.2, 0.1, 0.1, 0.2, 0.6, 0.3, 0.1, 0.2, 0.2, 0.2, 0.3, 0.1, 0.2, 0.1, 0.1, 0.1,
                   0.3, 0.1]


class Logger(Callback):

    def __init__(self, validation_data):
        super(Logger, self).__init__()
        self.X_val, self.y_true = validation_data
        self.metric1_array = []
        self.metric2_array = []
        self.jaccard_similarity = []
        self.train_loss = []
        self.val_loss = []

    def eval_jaccard_similarity(self):

        X_val = self.X_val
        y_true = self.y_true
        y_pred = self.model.predict(X_val)
        y_pred = np.array([[1 if y_pred[i, j] >= CLASS_THRESHOLD[j] else 0 for j in range(y_pred.shape[1])] for i in
                           range(len(y_pred))])
        return jaccard_similarity_score(y_true, y_pred)

    def eval_metrics(self):
        X_val = self.X_val
        y_true = self.y_true
        y_pred = self.model.predict(X_val)
        y_pred = np.array([[1 if y_pred[i, j] >= CLASS_THRESHOLD[j] else 0 for j in range(y_pred.shape[1])] for i in
                           range(len(y_pred))])
        correct_pred = 0
        sum_metric2 = 0
        for i in xrange(len(y_pred)):
            ind_pred = np.where(y_pred[i] == 1)[0]
            ind_true = np.where(y_true[i] == 1)[0]
            intersect = np.intersect1d(ind_pred, ind_true)
            if len(intersect) > 0:
                correct_pred += 1
            sum_metric2 += float(len(intersect)) / len(ind_true)
        return float(correct_pred) / len(y_true), sum_metric2 / len(y_true)

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs['loss'])

    def on_epoch_end(self, epoch, logs={}):

        score1, score2 = self.eval_metrics()
        jaccard = self.eval_jaccard_similarity()
        print "\n Accuracy1 for epoch %d is %f" % (epoch, score1)
        print "\n Accuracy2 for epoch %d is %f" % (epoch, score2)
        print "\n Jaccard similarity for epoch %d is %f" % (epoch, jaccard)

        self.metric1_array.append(score1)
        self.metric2_array.append(score2)
        self.jaccard_similarity.append(jaccard)
