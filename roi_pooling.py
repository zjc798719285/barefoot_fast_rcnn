import tensorflow as tf
import numpy as np
import math

def roi_layer(fc_map, out_size, rois):
    fc = []
    for roi_i in rois:
        fc_i = roi_pooling(fc_map, out_size, roi_i)
        fc = tf.concat(0, [fc, fc_i])
    return fc

def roi_pooling(fc_map, out_size, roi):
    # fc_map: feature map from a CNN net, which is a 4-dims tensor[1, height, width, channel]
    # out_size: the output size of feature map after roi_pooling, which is a list [height, width]
    # roi: [p_x, p_y, height, width], the resigon proposel
    roi_img = get_roi_img(fc_map, roi)
    roi_shape = roi_img.get_shape().as_list()
    st_x = math.ceil(roi_shape[1]/out_size[0])
    st_y = math.ceil(roi_shape[2]/out_size[1])
    roi_img = padding_img(roi_img, [st_x*out_size[0], st_y*out_size[1]])
    roi_fc_map = tf.nn.max_pool(roi_img, ksize=[1, st_x, st_y, 1], strides=[1, st_x, st_y, 1], padding='SAME')
    roi_fc = tf.reshape(roi_fc_map, [-1, out_size[0]*out_size[1]*roi_shape[3]])
    return roi_fc

def get_roi_img(fc_map, roi):
    # This function get the roi_img form feature map and roi label
    # return roi_img which is a tensor subsampled from fc_map, by using roi label
    shape_fc_map = fc_map.get_shape().as_list()
    roi_row = int(np.round(shape_fc_map[1] * roi[0]))
    roi_col = int(np.round(shape_fc_map[2] * roi[1]))
    roi_h = int(np.round(shape_fc_map[1] * roi[2]))
    roi_w = int(np.round(shape_fc_map[2] * roi[3]))
    roi_img = fc_map[:, roi_row:roi_row+roi_h, roi_col:roi_col+roi_w, :]
    return roi_img
def padding_img(img, shape):
    shape_img = img.get_shape().as_list()
    pad_rows = shape[0] - shape_img[1]
    pad_up = shape_img.copy(); pad_down = shape_img.copy()
    pad_down[1] = math.ceil(pad_rows / 2);pad_up[1] = math.floor(pad_rows / 2)
    img = tf.concat([tf.zeros(pad_up), img, tf.zeros(pad_down)], axis=1)
    shape_img = img.get_shape().as_list()
    pad_cols = shape[1] - shape_img[2]
    pad_left = shape_img.copy(); pad_right = shape_img.copy()
    pad_left[2] = math.ceil(pad_cols / 2);pad_right[2] = math.floor(pad_cols / 2)
    img = tf.concat([tf.zeros(pad_left), img, tf.zeros(pad_right)], axis=2)
    return img

if __name__ == '__main__':
    # unit testing interference


    map1 = tf.random_normal(shape=[1, 40, 80, 10], mean=0.0, stddev=1.0, dtype=tf.float32)
    # map1 = padding_img(map1, [41, 88])
    # print(map1)
    roi = [0.5, 0.5, 0.5, 0.5]
    roi_map = get_roi_img(map1, roi)
    roi_fc = roi_pooling(fc_map=map1, out_size=[5, 8], roi=roi)
    print(np.shape(roi_fc))
    print(np.shape(roi_map))



