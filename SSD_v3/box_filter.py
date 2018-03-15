import numpy as np
from ssd_box_encoder import iou_eval

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

# def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
# # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# # if there are no boxes, return an empty list
#     if len(boxes) == 0:
#         return []
#
# # grab the coordinates of the bounding boxes
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#
#     np.testing.assert_array_less(x1, x2)
#     np.testing.assert_array_less(y1, y2)
#
# # if the bounding boxes integers, convert them to floats --
# # this is important since we'll be doing a bunch of divisions
#     if boxes.dtype.kind == "i":
#         boxes = boxes.astype("float")
#
# # initialize the list of picked indexes
# 	pick = []
#
# # calculate the areas
#     area = (x2 - x1) * (y2 - y1)
#
# # sort the bounding boxes
#     idxs = np.argsort(probs)
#
# # keep looping while some indexes still remain in the indexes
# # list
#     while len(idxs) > 0:
# 		# grab the last index in the indexes list and add the
# 		# index value to the list of picked indexes
# 		last = len(idxs) - 1
# 		i = idxs[last]
# 		pick.append(i)
#
# 		# find the intersection
#
# 		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
# 		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
# 		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
# 		yy2_int = np.minimum(y2[i], y2[idxs[:last]])
#
# 		ww_int = np.maximum(0, xx2_int - xx1_int)
# 		hh_int = np.maximum(0, yy2_int - yy1_int)
#
# 		area_int = ww_int * hh_int
#
# 		# find the union
# 		area_union = area[i] + area[idxs[:last]] - area_int
#
# 		# compute the ratio of overlap
# 		overlap = area_int/(area_union + 1e-6)
#
# 		# delete all indexes from the index list that have
# 		idxs = np.delete(idxs, np.concatenate(([last],
# 			np.where(overlap > overlap_thresh)[0])))
#
# 		if len(pick) >= max_boxes:
# 			break
#
# # return only the bounding boxes that were picked using the integer data type
#     boxes = boxes[pick].astype("int")
#     probs = probs[pick]
#     return boxes, probs

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
            if classes_i[1] > classes_i[0] and classes_i[1] > 0.5:
                batch_classes.append(classes_i)
                batch_anchors.append(anchors_i)
                batch_offset.append(offset_i)
                rect = box_decoder2(anchor=anchors_i, offset=offset_i)
                batch_rect.append(rect)
        filted_classes.append(batch_classes)
        filted_anchors.append(batch_anchors)
        filted_offset.append(batch_offset)
        filted_rect.append(batch_rect)
    return filted_classes, filted_offset, filted_anchors, filted_rect

def rect_iou(roi_list, rect_batch):
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
        mean_iou_one_img = sum_iou_one_img / len_iou_one_img
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


# def class_pred_acc(cls_pred, cls_true):
#     (batch_size, boxes, classes) = np.shape(cls_pred)
#     n_pred = np.zeros([classes])
#     n_correct = np.zeros([classes])
#     index_pred = np.argmax(a=cls_pred, axis=2)
#     index_trued = np.argmax(a=cls_true, axis=2)
#     index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
#     index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
#     assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
#     for pred_i, true_i in zip(index_pred, index_trued):
#         for i in range(classes):
#             if pred_i == i:
#                 n_pred[i] = n_pred[i] + 1
#                 if true_i == i:
#                     n_correct[i] = n_correct[i] + 1
#
#     num_bk = 0; num_cls = 0
#     for i in range(batch_size):
#      for y_i in cls_true[i, :, :]:
#         if (y_i[0] == 1 and y_i[1] == 0):
#             num_bk += 1
#         if (y_i[0] == 0 and y_i[1] == 1):
#             num_cls += 1
#     num_hard = batch_size*boxes - num_bk - num_cls
#     return n_correct / n_pred, num_bk, num_cls, num_hard
def class_acc(_cls_pred, _cls_true):
    (batch_size, boxes, classes) = np.shape(_cls_pred)
    n_pos_pred = 0; n_pos_acc = 0; n_neg_pred = 0;n_neg_acc = 0
    n_pos = 0; n_neg = 0
    for i in range(batch_size):
        for j in range(boxes):
            cls_pred = _cls_pred[i, j, :]
            cls_true = _cls_true[i, j, :]
            if cls_pred[1] > cls_pred[0]:
                n_pos_pred += 1
            elif cls_pred[0] > cls_pred[1]:
                n_neg_pred += 1
            if cls_pred[1] > cls_pred[0] and cls_true[1] >= cls_true[0]:
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










