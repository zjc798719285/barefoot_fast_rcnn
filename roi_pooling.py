import tensorflow as tf

class RoiLayer(object):
    def __init__(self, out_size, rois):
        self.out_size = out_size
        self.rois = rois
    def __call__(self, fc_map):
        fc = []
        num_roi = self.rois.get_shape().as_list()[0]
        for i in range(num_roi):
            ind = tf.constant(value=i, shape=[1])
            roi_i = tf.reshape(tf.gather(indices=ind, params=self.rois), [4, -1])
            fc_i = ROI_pooling(fc_map, roi_i, self.out_size)
            fc.append(fc_i)
        final_fc = tf.concat(fc, axis=0)
        return final_fc

def ROI_pooling(fc_map, roi, out_size):
    # This function get the roi_img form feature map and roi label
    # return roi_img which is a tensor subsampled from fc_map, by using roi label
    shape_fc_map = fc_map.get_shape().as_list()
    roi_row = tf.cast(shape_fc_map[1] * tf.gather(indices=0, params=roi)[0], tf.int32)
    roi_col = tf.cast(shape_fc_map[2] * tf.gather(indices=1, params=roi)[0], tf.int32)
    roi_h = tf.cast(shape_fc_map[1] * tf.gather(indices=2, params=roi)[0], tf.int32)   # have some bugs
    roi_w = tf.cast(shape_fc_map[2] * tf.gather(indices=3, params=roi)[0], tf.int32)   #
    roi_img = fc_map[:, roi_row:roi_row+roi_h, roi_col:roi_col+roi_w, :]
    shape = [int(x) for x in out_size]
    roi_img = tf.image.resize_images(roi_img, tuple(shape))
    return roi_img


if __name__ == '__main__':
    # unit testing interference
    size = [14.0, 14.0]
    size1 = [int(x) for x in size]
    shape = tuple(size1)
    print(size)
