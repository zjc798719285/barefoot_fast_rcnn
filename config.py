import tensorflow as tf
config = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    'resize_scale', 200, 'input a float')
tf.app.flags.DEFINE_integer(
    'neg_pos_ratio', 3, 'input a integer')
tf.app.flags.DEFINE_float(
    'neg_loss_scales', 2, 'input a float')
tf.app.flags.DEFINE_integer(
    'num_classes', 20, 'input a integer')
tf.app.flags.DEFINE_float(
    'overlap_thresh', 0.7, 'input a float')
tf.app.flags.DEFINE_float(
    'num_boxes', 300, 'input a integer')

