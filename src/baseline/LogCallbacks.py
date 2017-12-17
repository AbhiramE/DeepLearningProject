from keras import backend as K
import numpy as np
from keras.callbacks import Callback

K = 3


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
        accuracy = 0.0
        precision = 0.0
        for i in xrange(len(y_pred)):
            pred_indices = np.argsort(y_pred[i])[-K:][::-1]
            true_indices = np.argsort(y_true[i])[-K:][::-1]
            intersect = np.intersect1d(pred_indices, true_indices)
            if len(intersect) > 0:
                accuracy += 1.0
            precision += float(len(intersect)) / K
        return accuracy / len(y_true), precision / len(y_true)

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs['loss'])

    def on_epoch_end(self, epoch, logs={}):
        score1, score2 = self.eval_metrics()
        jaccard = self.eval_jaccard_similarity()
        print "\n Accuracy for epoch %d is %f" % (epoch, score1)
        print "\n Precision for epoch %d is %f" % (epoch, score2)
        print "\n Jaccard similarity for epoch %d is %f" % (epoch, jaccard)

        self.metric1_array.append(score1)
        self.metric2_array.append(score2)
        self.jaccard_similarity.append(jaccard)
