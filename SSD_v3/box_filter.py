import numpy as np
from ssd_box_encoder import iou_eval

def box_filter(pred_offset, pred_classes, pred_anchors, num_positives = 1):
    pred_rect_list = [];pred_anchors_list = [];pred_offset_list = []
    rect = np.zeros([4])
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    for i in range(batch_size):
        classes_ind = np.argsort(a=pred_classes[i, :, 1], axis=0)
       # print('classe=**********', pred_classes[i, classes_ind[-num_positives], :])
        # max_confidence = pred_classes[i, classes_ind, 1]
        offset = pred_offset[i, classes_ind[-num_positives], :]
        anchors = pred_anchors[i, classes_ind[-num_positives], :]
        rect[0] = anchors[0] + offset[0]
        rect[1] = anchors[1] + offset[1]
        rect[2] = anchors[2] * offset[2]
        rect[3] = anchors[3] * offset[3]
        pred_rect_list.append(rect)
        pred_anchors_list.append(anchors)
        pred_offset_list.append(offset)
    return np.array(pred_rect_list),\
            np.array(pred_anchors_list),\
            np.array(pred_offset_list)
def rect_pred(pred_offset, pred_anchors):
    rect = np.copy(pred_offset).astype(np.float)
    # rect[...,0] =


def box_filter2(pred_offset, pred_classes, pred_anchors, num_positives = 1):
    pred_rect_list = [];pred_anchors_list = [];pred_offset_list = []
    rect = np.zeros([4])
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    for i in range(batch_size):
        red_offset_list = []
    return

def batch_mean_iou(roi_list, rect):
    sum_iou = 0
    for roi_i, rect_i in zip(roi_list, rect):
        iou = iou_eval(dr=rect_i, gt=roi_i)
        sum_iou += iou
    return sum_iou/len(roi_list)













def class_pred_acc(cls_pred, cls_true):
    (batch_size, boxes, classes) = np.shape(cls_pred)
    n_pred = np.zeros([classes])
    n_correct = np.zeros([classes])
    index_pred = np.argmax(a=cls_pred, axis=2)
    index_trued = np.argmax(a=cls_true, axis=2)
    index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
    index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
    assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
    for pred_i, true_i in zip(index_pred, index_trued):
        for i in range(classes):
            if pred_i == i:
                n_pred[i] = n_pred[i] + 1
                if true_i == i:
                    n_correct[i] = n_correct[i] + 1

    num_bk = 0; num_cls = 0
    for i in range(batch_size):
     for y_i in cls_true[i, :, :]:
        if (y_i[0] == 1 and y_i[1] == 0):
            num_bk += 1
        if (y_i[0] == 0 and y_i[1] == 1):
            num_cls += 1
    num_hard = batch_size*boxes - num_bk - num_cls
    return n_correct / n_pred, num_bk, num_cls, num_hard



def class_pred_acc2(cls_pred, cls_true):
    (batch_size, boxes, classes) = np.shape(cls_pred)
    n_pred = np.zeros([classes])
    n_pred_true = np.zeros([classes])
    n_true = np.array([batch_size * boxes - np.sum(cls_true[:, :, 1]), np.sum(cls_true[:, :, 1])])
    num_pos = 0; num_neg = 0
    for i in range(batch_size):
        for j in range(boxes):
            if cls_true[i, j, 1] == 1:
                num_pos += 1
            elif cls_true[i, j, 0] == 1:
                num_neg += 1
            if cls_pred[i, j, 1] > cls_pred[i, j, 0]:
                n_pred[1] += 1
                if cls_true[i, j, 0] == 0:
                    n_pred_true[1] += 1
            else:
                n_pred[0] += 1
                if cls_true[i, j, 0] == 1:
                    n_pred_true[0] += 1
    num_hard = batch_size * boxes - num_pos - num_neg
    acc = n_pred_true / n_pred
    recall = n_pred_true / np.array([num_neg, num_pos + num_hard])
    return acc, recall, num_pos, num_hard, num_neg,








    for i in range(batch_size):
        for y_i in cls_true[i, :, :]:
            if (y_i[0] == 1 and y_i[1] == 0):
                num_bk += 1
            if (y_i[0] == 0 and y_i[1] == 1):
                num_cls += 1
    num_hard = batch_size * boxes - num_bk - num_cls

    return n_correct / n_pred, num_bk, num_cls, num_hard


