import tensorflow as tf
class RoiPooling(object):
    def __init__(self, pool_size, num_rois):
        self.pool_size = pool_size
        self.num_rois = num_rois


    def __call__(self, base_layer, rois):
        pool_layer_list = []
        for idx in range(self.num_rois):
            (batch_size, height, width, channel) = base_layer.get_shape()
            s1 = width * rois[idx, 0]
            x = tf.cast(s1, tf.int32)
            y = tf.cast(height * rois[idx, 1], tf.int32)
            w = tf.cast(width * rois[idx, 2], tf.int32)
            h = tf.cast(height * rois[idx, 3], tf.int32)
            pool_layer = tf.image.resize_images(images=base_layer[:, y:y+h, x:x+w, :],
                                                size=(self.pool_size, self.pool_size))
            pool_layer_list.append(pool_layer)
        pool_layer = tf.concat(values=pool_layer_list, axis=0)
        output = tf.reshape(tensor=pool_layer, shape=[self.num_rois, self.pool_size, self.pool_size, channel])
        return output  #tensor:[num_rois, pool_size, pool_size, channel]
















