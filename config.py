import tensorflow as tf

flags = tf.app.flags
############################
#    hyper parameters      #
############################
# for training
flags.DEFINE_integer('batch_shape', [1, 128, 59, 3], 'Shape of a batch tensor')
flags.DEFINE_integer('num_rois', 20, 'Num rois for per image')
flags.DEFINE_integer('roi_shape', [4, 2], 'Shape of feature map after roi-pooling')
flags.DEFINE_integer('epoch', 10, 'Max number of epoch')

flags.DEFINE_float('learning_rate', 0.1, 'learning_rate')
flags.DEFINE_float('rho', 0.9, 'rho')

flags.DEFINE_string('train_path', 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
                    , 'train_path')
flags.DEFINE_string('test_path', 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
                    , 'test_path')


cfg = tf.app.flags.FLAGS