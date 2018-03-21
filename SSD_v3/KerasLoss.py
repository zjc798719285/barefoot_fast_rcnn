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
    loss_L1 = tf.abs(anchor_pred - anchor_true)
    loss_L2 = 0.5 * (anchor_pred - anchor_true)**2
    loss = tf.reduce_mean(tf.where(tf.less(loss_L1, 1.0), loss_L2, loss_L1 - 0.5))
    # loss = tf.reduce_mean(loss)
    return loss

def cls_loss(y_pred, y_true):
    pos_neg_ratio = 1
    classification_loss = log_loss(y_pred=y_pred, y_true=y_true)
    pos_mask = y_true[:, :, 1]; neg_mask = y_true[:, :, 0]
    num_pos = tf.reduce_sum(pos_mask); num_neg = tf.reduce_sum(neg_mask)
    pos_loss = tf.reduce_sum(classification_loss * pos_mask)
    neg_loss_all = classification_loss * neg_mask
    num_neg_keep = tf.cast(tf.minimum(tf.maximum(pos_neg_ratio * num_pos, 1), num_neg), tf.int32)  #边界限定
    neg_loss_all_1D = tf.reshape(neg_loss_all, [-1])  # Tensor of shape (batch_size * n_boxes,)
    values, indices = tf.nn.top_k(neg_loss_all_1D, num_neg_keep, False)
    neg_loss = tf.reduce_sum(values)
    class_loss = (pos_loss + neg_loss)/(num_pos + num_neg)
    return class_loss

def loc_loss(y_pred, y_true):
    loc_loss = smooth_L1(anchor_pred=y_pred, anchor_true=y_true)
    pos_mask = y_true[:, :, 1]; num_pos = tf.reduce_sum(pos_mask)
    loc_loss_pos = tf.reduce_sum(loc_loss * pos_mask) / num_pos
    return loc_loss_pos
def loss(y_pred, y_true):
    y_pred_cls = y_pred[:, :, 0:3]
    y_true_cls = y_true[:, :, 0:3]
    y_pred_loc = y_pred[:, :, 4:7]
    y_true_loc = y_true[:, :, 4:7]
    loss_cls = cls_loss(y_pred=y_pred_cls, y_true=y_true_cls)
    loss_loc = loc_loss(y_pred=y_pred_loc, y_true=y_true_loc)
    loss = 0.1 * loss_cls + loss_loc
    return loss