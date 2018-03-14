import tensorflow as tf

# def cls_loss(y_pred, y_true):
#     # y_pred = tf.maximum(y_pred, 1e-15)
#     # cls_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
#     cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
#     return cls_loss

def log_loss(y_pred, y_true):
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(y_pred, 1e-15)
    one = tf.constant([1.0], dtype=tf.float32)
    cross_entropy = y_true * tf.log(y_pred) + (one - y_true) * tf.log(one - y_pred)
    loss = -tf.reduce_sum(cross_entropy, axis=2)
    return loss

def smooth_L1(anchor_pred, anchor_true):
    # loss_L1 = tf.abs(anchor_pred - anchor_true)
    loss_L2 = 0.5 * (anchor_pred - anchor_true)**2
    # loss = tf.reduce_mean(tf.where(tf.less(loss_L1, 1.0), loss_L2, loss_L1 - 0.5))
    loss = tf.reduce_mean(loss_L2)
    return loss

def cls_loc_loss(anchor_pred, anchor_true, y_pred, y_true,pos_neg_ratio):
    loc_loss = smooth_L1(anchor_pred=anchor_pred, anchor_true=anchor_true)
    classification_loss = log_loss(y_pred=y_pred, y_true=y_true)
    pos_mask = y_true[:, :, 1]; neg_mask = y_true[:, :, 0]
    num_pos = tf.reduce_sum(pos_mask); num_neg = tf.reduce_sum(neg_mask)
    # if tf.less(num_pos, tf.constant(1.01)):
    #     raise ValueError('No positive anchors are selected')
    pos_loss = tf.reduce_mean(classification_loss * pos_mask)
    neg_loss_all = classification_loss * neg_mask
    num_neg_keep = tf.cast(tf.minimum(tf.maximum(pos_neg_ratio * num_pos, 1), num_neg), tf.int32)  #边界限定
    neg_loss_all_1D = tf.reshape(neg_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
    # ...and then we get the indices for the `n_negative_keep` boxes with the highest loss out of those...
    values, indices = tf.nn.top_k(neg_loss_all_1D, num_neg_keep, False)
    neg_loss = tf.reduce_mean(values)
    class_loss = pos_loss + neg_loss

    return loc_loss, class_loss