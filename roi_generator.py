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
    rois = []; cls_label = []
    border_top = roi[0]; border_bottom = 1 - roi[0] - roi[2]
    border_left = roi[1]; border_right = 1 - roi[1] - roi[3]
    for i in range(num_pair):
        # negtive roi and label generation
        x_offset = np.random.uniform(low=-border_top, high=border_bottom)
        y_offset = np.random.uniform(low=-border_left, high=border_right)
        h_zoom = np.random.uniform(low=0.5, high=1.3)
        w_zoom = np.random.uniform(low=0.5, high=1.3)
        roi_neg = [roi[0] + x_offset, roi[1] + y_offset, roi[2] * h_zoom, roi[3] * w_zoom]
        rois.append(roi_neg); cls_label.append([0, 1])
        # positive roi and label generation
        h_expand = np.random.uniform(low=1, high=1.1)
        w_expand = np.random.uniform(low=1, high=1.1)
        x_up_off = np.random.uniform(low=-(h_expand-1)*roi[2], high=0)
        y_left_off = np.random.uniform(low=-(w_expand-1)*roi[3], high=0)
        roi_pos = [roi[0] + x_up_off, roi[1] + y_left_off, roi[2] * h_expand, roi[3] * w_expand]
        rois.append(roi_pos); cls_label.append([1, 0])
    return rois, cls_label

if __name__ == '__main__':
   roi = [0.1, 0.3, 0.4, 0.6]
   rois, label = pos_neg_roi_generator(roi, 10)
   print(rois)
   print(label)

