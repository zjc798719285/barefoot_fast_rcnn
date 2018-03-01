import numpy as np
from ssd_box_encoder import iou_eval

def box_filter(pred_offset, pred_classes, pred_anchors, num_positives = 1):
    pred_rect_list = [];pred_anchors_list = [];pred_offset_list = []
    rect = np.zeros([4])
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    for i in range(batch_size):
        classes_ind = np.argsort(a=pred_classes[i, :, 1], axis=0)
        print('classe=**********', pred_classes[i, classes_ind[-num_positives], :])
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
    num_bk = 0; num_correct_bk = 0; num_cls = 0; num_correct_cls = 0; num_bk_to_cls = 0
    index_pred = np.argmax(a=cls_pred, axis=2)
    index_trued = np.argmax(a=cls_true, axis=2)
    index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
    index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
    assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
    for pred_i, true_i in zip(index_pred, index_trued):
        if true_i == 0:
            num_bk += 1
            if pred_i == 0:
                num_correct_bk += 1
            if pred_i == 1:
                num_bk_to_cls += 1
        if true_i == 1:
            num_cls += 1
            if pred_i == 1:
                num_correct_cls += 1
    return num_correct_bk/num_bk, \
           num_correct_cls/(num_cls + 10e-6), \
           np.sum(index_pred), \
           np.sum(index_trued), \
           num_bk_to_cls/(num_correct_cls + num_bk_to_cls)


def class_pred_acc2(cls_pred, cls_true):
    (batch_size, boxes, classes) = np.shape(cls_pred)
    num_bk = 0; num_correct_bk = 0; num_cls = 0; num_correct_cls = 0; num_bk_to_cls = 0
    np.where()
    index_pred = np.argmax(a=cls_pred, axis=2)
    index_trued = np.argmax(a=cls_true, axis=2)
    index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
    index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
    assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
    for pred_i, true_i in zip(index_pred, index_trued):
        if true_i == 0:
            num_bk += 1
            if pred_i == 0:
                num_correct_bk += 1
            if pred_i == 1:
                num_bk_to_cls += 1
        if true_i == 1:
            num_cls += 1
            if pred_i == 1:
                num_correct_cls += 1
    return num_correct_bk/num_bk, \
           num_correct_cls/(num_cls + 10e-6), \
           np.sum(index_pred), \
           np.sum(index_trued), \
           num_bk_to_cls/(num_correct_cls + num_bk_to_cls)



