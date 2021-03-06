import tensorflow as tf

def loss_box_regressor(gt, rois, dr, mode):
    if mode == 'abs':
       loss = tf.reduce_mean(tf.abs(tf.subtract(tf.subtract(gt, rois), dr)))
       return loss
    if mode == 'L2':
       loss = tf.reduce_mean(tf.square(tf.subtract(tf.subtract(gt, rois), dr)))
       return loss


def loss_classify(cls_predic, labels):
    cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=cls_predic))
    return cls_loss


def loss_RPN(RPN_rois, gt, num_rois, mode):
    loss = tf.zeros(shape=[1], dtype=tf.float32, name=None)
    for index in range(num_rois):
        RPN_rois_i = tf.reshape(tf.gather(indices=index, params=RPN_rois), [-1, 4])
        gt_i = gt[index, :]
        loss = tf.reduce_mean(tf.abs(tf.subtract(gt_i, RPN_rois_i)))
      #  loss = tf.add(loss, loss_box_regressor(gt_i, RPN_rois_i, mode))
    return tf.div(loss, num_rois)
