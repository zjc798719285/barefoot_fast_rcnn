import tensorflow as tf

flags = tf.app.flags
############################
#    hyper parameters      #
############################
# for training
flags.DEFINE_integer('batch_shape', [1, 128, 59, 3], 'batch_shape')
flags.DEFINE_integer('num_rois', 20, 'num_rois_per_img')
flags.DEFINE_integer('roi_shape', [4, 2], 'roi_shape')
flags.DEFINE_integer('epoch', 10, 'epoch')


cfg = tf.app.flags.FLAGS