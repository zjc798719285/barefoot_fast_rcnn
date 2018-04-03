import tensorflow as tf


def loss_rpn_cls(y_pred, y_true):
    neg_pos_ratio = 2; c = 2
    y_pred = tf.maximum(y_pred, 1e-15)
    cross_entropy = -tf.reduce_sum(y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred), axis=2)
    pos_mask = tf.reduce_sum(y_true[:, :, 1:-1], axis=-1) #此处pos_mask一定要注意，
    neg_mask = y_true[:, :, 0]
    num_pos = tf.reduce_sum(pos_mask); num_neg = tf.reduce_sum(neg_mask)
    loss_pos = tf.reduce_sum(cross_entropy * pos_mask) / num_pos
    loss_neg_all = cross_entropy * neg_mask
    num_neg_keep = tf.cast(tf.minimum(tf.maximum(num_pos * neg_pos_ratio, 1), num_neg), tf.int32)  #边界限定
    loss_neg_1D = tf.reshape(loss_neg_all, [-1])
    values, _ = tf.nn.top_k(loss_neg_1D, num_neg_keep, False)
    loss_neg = tf.reduce_mean(values)
    loss = loss_pos + c * loss_neg
    return loss


def loss_rpn_regress(y_pred, y_true):
    loss_L1 = tf.abs(y_pred - y_true)
    loss_L2 = 0.5 * (y_pred - y_true) ** 2
    loss = tf.reduce_mean(tf.where(tf.less(loss_L1, 1.0), loss_L2, loss_L1 - 0.5))
    return loss