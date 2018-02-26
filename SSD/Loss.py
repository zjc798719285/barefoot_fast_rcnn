import tensorflow as tf

def cls_loss(y_pred, y_true):
    cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    return cls_loss

def smooth_L1(anchor_pred, anchor_true):
    s1 = tf.reduce_mean((anchor_pred[:, :, 0] - anchor_true[:, :, 0])**2)
    s2 = tf.reduce_mean((anchor_pred[:, :, 1] - anchor_true[:, :, 1])**2)
    s3 = tf.reduce_mean((tf.log(tf.div(anchor_true[:, :, 2], anchor_pred[:, :, 2]))))
    s4 = tf.reduce_mean(tf.log(tf.div(anchor_true[:, :, 3], anchor_pred[:, :, 3])))
    offset_loss = s1 + s2 + s3 + s4
    return offset_loss
