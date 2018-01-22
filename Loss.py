import tensorflow as tf

def loss_box_regressor(gt, dr, mode):
    if mode == 'abs':
       loss = tf.reduce_mean(tf.abs(tf.subtract(gt, dr)))
       return loss
    if mode == 'L2':
       loss = tf.reduce_mean(tf.square(tf.subtract(gt, dr)))
       return loss

def loss_classify(cls_predic, labels):
    cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=cls_predic))
    return cls_loss
