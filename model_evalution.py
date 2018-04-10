import numpy as np
from config import config as C
class ModelEval(object):


    def __call__(self, cls_pred, cls_true, num_classes=C.num_classes):
        num_classes += 1
        idx_pred = np.argmax(cls_pred, axis=1)
        idx_true = np.argmax(cls_true[0, :, :], axis=1)
        n_pred = np.zeros(num_classes);n_true = np.zeros(num_classes)
        n_correct = np.zeros(num_classes)
        for pred_i, true_i in zip(idx_pred, idx_true):
            n_pred[pred_i] += 1
            n_true[true_i] += 1
            if pred_i == true_i:
                n_correct[pred_i] += 1
        acc_neg = n_correct[0] / n_pred[0]
        acc_pos = np.sum(n_correct[1:-1]) / np.sum(n_pred[1:-1])
        recall_pos = n_correct[0] / n_true[0]
        recall_neg = np.sum(n_correct[1:-1]) / np.sum(n_pred[1:-1])
        return [acc_neg, acc_pos], [recall_neg, recall_pos]







