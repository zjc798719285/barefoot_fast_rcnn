import tensorflow as tf
import numpy as np


def pos_neg_roi_generator(roi, num_pair):
    # This function used for generate positive roi and negtive roi for one image,and return rois and labels
    # The rois and label used for training classification network
    # Positive roi are labeled [1, 0],while negtive roi are labeled [0, 1]
    # The positive roi and negtive roi are generate alternately in a loop, so we can get 2*num_pair rois-
    # -after using this function
    # roi: The labeled roi for one image which is shaped [p_x, p_y, height, width]
    # num_pair: How many rois we except to generate
    rois = []; cls_label = [];
    border_top = roi[0]; border_bottom = 1 - roi[0] - roi[2]
    border_left = roi[1]; border_right = 1 - roi[1] - roi[3]
    for i in range(num_pair):
        # negtive roi and label generation
        x_offset = np.random.uniform(low=-border_top, high=border_bottom)
        y_offset = np.random.uniform(low=-border_left, high=border_right)
        h_zoom = np.random.uniform(low=0.5, high=1.3)  #set parameters
        w_zoom = np.random.uniform(low=0.5, high=1.3)
        roi_neg = [roi[0] + x_offset, roi[1] + y_offset, roi[2] * h_zoom, roi[3] * w_zoom]
        rois.append(roi_neg); cls_label.append([0, 1])
        # positive roi and label generation
        h_expand = np.random.uniform(low=1, high=1.1)  #set high parameter
        w_expand = np.random.uniform(low=1, high=1.1)
        x_up_off = np.random.uniform(low=-(h_expand-1)*roi[2], high=0)
        y_left_off = np.random.uniform(low=-(w_expand-1)*roi[3], high=0)
        roi_pos = [roi[0] + x_up_off, roi[1] + y_left_off, roi[2] * h_expand, roi[3] * w_expand]
        rois.append(roi_pos); cls_label.append([1, 0])
    return rois, cls_label

def box_roi_generator(cls_label, roi):
    box_roi = []
    for cls_i in cls_label:
      if cls_i  == [1, 0]:
          box_roi.append(roi)
      elif cls_i  == [0, 1]:
          box_roi.append([0, 0, 0, 0])
    return box_roi


def iou_eval(gt, dr):
    # gt: GroundTruth roi
    # dr: DetectionResult roi
    # return overlap ratio of gt and dr
 start_x_gt = gt[0]; end_x_gt = gt[0] + gt[2]
 start_y_gt = gt[1]; end_y_gt = gt[1] + gt[3]
 start_x_dr = dr[0]; end_x_dr = dr[0] + dr[2]
 start_y_dr = dr[1]; end_y_dr = dr[1] + dr[3]
 if (end_x_gt <= start_x_dr or end_x_dr <= start_x_gt or
     end_y_gt <= start_y_dr or end_y_dr <= start_y_gt):
     return 0
 else:
     gt_area = gt[2] * gt[3]; dr_area = dr[2] * dr[3]
     start_x = max(start_x_gt, start_x_dr); end_x = min(end_x_gt, end_x_dr)
     start_y = max(start_y_gt, start_y_dr); end_y = min(end_y_gt, end_y_dr)
     h_inter = end_x - start_x; w_inter = end_y - start_y
     inter_area = h_inter * w_inter
     union_area = gt_area + dr_area - inter_area
     return  inter_area / union_area

def loss_box(gt, dr):
     loss = tf.reduce_mean(tf.abs(tf.subtract(gt, dr)))
     return loss


if __name__ == '__main__':
   # roi1 = [0.1, 0.1, 0.2, 0.2]
   # roi2 = [0.2, 0.2, 0.2, 0.2]
   # roi3 = [0.3, 0.2, 0.2, 0.2]
   # roi4 = [0.2, 0.3, 0.2, 0.2]
   # print(iou_eval(roi1, roi2))
   # print(iou_eval(roi1, roi3))
   # print(iou_eval(roi1, roi4))

   roi = [0.2, 0.1, 0.5, 0.5]
   rois, label = pos_neg_roi_generator(roi, 10)
   box_roi = box_roi_generator(cls_label=label, roi=roi)

   for box_roi_i in box_roi:
       print(box_roi_i)
   print(np.shape(rois))
   print(np.shape(label))

