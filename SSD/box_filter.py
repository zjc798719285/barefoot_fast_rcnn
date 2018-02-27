import numpy as np


def box_filter(pred_offset, pred_classes):
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    for i in range(batch_size):
        for j in range(num_boxes):
            classes = pred_classes[i, j, :]




    return


def class_pred_acc(cls_pred, cls_true):
    (batch_size, boxes, classes) = np.shape(cls_pred)
    num_bk = 0; cor_bk = 0; num_cls = 0; cor_cls = 0
    index_pred = np.argmax(a=cls_pred, axis=2)
    index_trued = np.argmax(a=cls_true, axis=2)
    index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
    index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
    assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
    for pred_i, true_i in zip(index_pred, index_trued):
        if true_i == 0:
            num_bk += 1
            if pred_i == 0:
                cor_bk += 1
        if true_i == 1:
            num_cls += 1
            if cor_cls == 1:
                cor_cls += 1
    return cor_bk/num_bk, cor_cls/num_cls




