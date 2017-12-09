from keras import backend as K
import numpy as np
from keras.callbacks import Callback


class Logger(Callback):

    def __init__(self, validation_data):
        super(Logger, self).__init__()
        self.X_val, self.y_true = validation_data
        self.metric_array = []
        self.jaccard_similarity = []

    def eval_jaccard_similarity(self):

        X_val = self.X_val
        y_true = self.y_true
        y_pred = self.model.predict(X_val)

        numerator = np.sum(np.minimum(y_true, y_pred))
        denominator = np.sum(np.maximum(y_true, y_pred))
        return numerator/denominator


    def eval_metric(self):
        X_val = self.X_val
        y_true = self.y_true
        y_pred = self.model.predict(X_val)
        y_true_cols = np.count_nonzero(y_true, axis=1)
        correct_pred = 0
        for i in xrange(len(y_pred)):
            pred_indices = np.argsort(y_pred[i])[-y_true_cols[i]:][::-1]
            true_indices = np.argsort(y_true[i])[-y_true_cols[i]:][::-1]
            if len(np.intersect1d(pred_indices, true_indices)) > 0:
                correct_pred += 1
        return float(correct_pred) / len(X_val)

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_metric()
        jaccard = self.eval_jaccard_similarity()
        print "\n Accuracy for epoch %d is %f" % (epoch, score)
        print "\n Jaccard similarity for epoch %d is %f" % (epoch, score)
        self.metric_array.append(score)
        self.jaccard_similarity.append(jaccard)
