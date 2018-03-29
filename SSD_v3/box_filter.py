import numpy as np
from ssd_box_encoder import iou_eval
import time

def box_decoder(anchor, offset):
    anchor_x = anchor[0]; anchor_y = anchor[1]
    anchor_w = anchor[2]; anchor_h = anchor[3]
    offset_x = offset[0]; offset_y = offset[1]
    offset_w = offset[2]; offset_h = offset[3]
    rect_x = offset_x * anchor_h + anchor_x
    rect_y = offset_y * anchor_w + anchor_y
    rect_w = anchor_w * np.exp(offset_w)
    rect_h = anchor_h * np.exp(offset_h)
    rect = [rect_x, rect_y, rect_w, rect_h]
    return rect
def box_decoder2(anchor, offset):
    anchor_x = anchor[0]; anchor_y = anchor[1]
    anchor_w = anchor[2]; anchor_h = anchor[3]
    offset_x = offset[0]; offset_y = offset[1]
    offset_w = offset[2]; offset_h = offset[3]
    rect_x = anchor_x + offset_x
    rect_y = anchor_y + offset_y
    rect_w = anchor_w + offset_w
    rect_h = anchor_h + offset_h
    rect = [rect_x, rect_y, rect_w, rect_h]
    return rect


def box_decoder3(anchor, offset):
    anchor_x = anchor[0]; anchor_y = anchor[1]
    anchor_w = anchor[2]; anchor_h = anchor[3]
    offset_x = offset[0]; offset_y = offset[1]
    offset_w = offset[2]; offset_h = offset[3]
    rect_x = anchor_x - np.log(1/offset_x - 1)
    rect_y = anchor_y - np.log(1/offset_y - 1)
    rect_w = anchor_w - np.log(1/offset_w - 1)
    rect_h = anchor_h - np.log(1/offset_h - 1)
    rect = [rect_x, rect_y, rect_w, rect_h]
    return rect

def NMS(rect, classes, threshold, max_boxes = 100):
    t1 = time.time()
    rect = np.array(rect)
    classes = np.array(classes)
    boxes = np.concatenate((rect, classes), 1)
    pick_boxes = []
    box = boxes[boxes[:, 5].argsort()]; boxes = box.tolist()
    while boxes:
        max_box = boxes[-1]
        pick_boxes.append(max_box[0:4])
        boxes.remove(max_box)
        for box_i in boxes:
            if iou_eval(gt=box_i[0:4], dr=max_box[0:4]) > threshold:
                boxes.remove(box_i)
    if len(pick_boxes) > max_boxes:
        pick_boxes = pick_boxes[0:max_boxes]
    t2 = time.time()
    return pick_boxes, t2 - t1

def box_filter(pred_offset, pred_classes, pred_anchors):
    filted_classes =[]; filted_anchors = []; filted_offset = []
    filted_rect = []
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    for i in range(batch_size):
        offset = pred_offset[i, :, :]; classes = pred_classes[i, :, :]
        anchors = pred_anchors[i, :, :]
        batch_classes = []; batch_anchors = []; batch_offset=[]
        batch_rect = []
        for offset_i, classes_i, anchors_i in zip(offset, classes, anchors):
            if classes_i[1] > classes_i[0]:
                batch_classes.append(classes_i)
                batch_anchors.append(anchors_i)
                batch_offset.append(offset_i)
                rect = box_decoder2(anchor=anchors_i, offset=offset_i)
                batch_rect.append(rect)
        # rect, time_rect = NMS(rect=batch_rect, classes=batch_classes, threshold=0.7, max_boxes=300)
        # anchors, time_anchors = NMS(rect=batch_anchors, classes=batch_classes, threshold=0.7, max_boxes=300)
        # print('time_rect:', time_rect, 'time_anchors:', time_anchors)
        filted_classes.append(batch_classes)
        filted_anchors.append(batch_anchors)
        filted_offset.append(batch_offset)
        filted_rect.append(batch_rect)
    return filted_classes, filted_offset, filted_anchors, filted_rect

def rect_iou(roi_list, rect_batch):
    eps = 10e-6
    batch_size = np.shape(rect_batch)[0]
    sum_iou = 0; mean_iou_one_img = 0
    for i in range(batch_size):
        roi = roi_list[i][0]
        rect = rect_batch[i]
        sum_iou_one_img = 0; len_iou_one_img = 0
        for rect_i in rect:
            len_iou_one_img += 1
            iou = iou_eval(gt=roi, dr=rect_i)
            sum_iou_one_img += iou
        mean_iou_one_img = sum_iou_one_img / (len_iou_one_img+eps)
        sum_iou += mean_iou_one_img
    mean_iou = sum_iou / batch_size
    return mean_iou

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



def class_acc(_cls_pred, _cls_true):
    eps = 10e-6
    (batch_size, boxes, classes) = np.shape(_cls_pred)
    n_pos_pred = eps; n_pos_acc = eps; n_neg_pred = eps;n_neg_acc = eps
    n_pos = eps; n_neg = eps
    for i in range(batch_size):
        for j in range(boxes):
            cls_pred = _cls_pred[i, j, :]
            cls_true = _cls_true[i, j, :]
            if cls_pred[1] > cls_pred[0]:
                n_pos_pred += 1
            elif cls_pred[0] > cls_pred[1]:
                n_neg_pred += 1
            if cls_pred[1] > cls_pred[0] and cls_true[1] > cls_true[0]:
                n_pos_acc += 1
            elif cls_pred[0] > cls_pred[1] and cls_true[0] > cls_true[1]:
                n_neg_acc += 1
            if cls_true[1] > cls_true[0]:
                n_pos += 1
            if cls_true[0] > cls_true[1]:
                n_neg += 1
    acc = np.array([n_neg_acc, n_pos_acc]) / np.array([n_neg_pred, n_pos_pred])
    recall = np.array([n_neg_acc, n_pos_acc]) / np.array([n_neg, n_pos])
    n_hard = batch_size * boxes - n_pos - n_neg
    return acc, recall, n_pos, n_neg, n_hard



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










