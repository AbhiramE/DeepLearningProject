from keras import backend as K
import numpy as np
from keras.callbacks import Callback


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

        numerator = np.sum(np.minimum(y_true, y_pred))
        denominator = np.sum(np.maximum(y_true, y_pred))
        return numerator / denominator

    def eval_metrics(self):
        X_val = self.X_val
        y_true = self.y_true
        y_pred = self.model.predict(X_val)
        y_true_cols = np.count_nonzero(y_true, axis=1)
        correct_pred = 0
        sum_metric2 = 0.0
        for i in xrange(len(y_pred)):
            pred_indices = np.argsort(y_pred[i])[-y_true_cols[i]:][::-1]
            true_indices = np.argsort(y_true[i])[-y_true_cols[i]:][::-1]
            intersect = np.intersect1d(pred_indices, true_indices)
            if len(intersect) > 0:
                correct_pred += 1
            sum_metric2 += float(len(intersect)) / y_true_cols[i]
        return float(correct_pred) / len(X_val), sum_metric2 / len(y_true)

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
