from SSDModel import SSDModel
import tensorflow as tf
import numpy as np
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
import Loss

######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
batch_size = 32
#############
# Load Data #
#############


TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, 408, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, 408, 2])
classes, offset, anchors = SSDModel(l2_regularization=0,
                                    n_classes=1,
                                    aspect_ratios=[0.5, 1, 2])(TRAIN_X)
loss_cls = Loss.cls_loss(y_pred=classes, y_true=TRAIN_CLASSES)
loss_L1 = Loss.smooth_L1(anchor_pred=offset, anchor_true=TRAIN_ANCHORS)
loss = loss_cls + loss_L1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=32)
    testData = BatchGenerator(image=test_x, roi=test_roi, classes=test_cls, batch_size=32)
    train_x, train_roi_list, train_class_list = trainData.next_batch()
    classes2, offset2, anchors2 = sess.run([classes, offset, anchors], feed_dict={TRAIN_X: train_x})
    for i in range(100):
        print(i)
        train_x, train_roi_list, train_class_list = trainData.next_batch()
        y_classes, y_anchors = ssd_box_encoder_batch(roi_list=train_roi_list,
                                                     classes_list=train_class_list,
                                                     anchors=anchors2,
                                                     iou_threshold=0.5,
                                                     num_classes=1)
        classes1, offset1, anchors1, loss_cls1, loss_L11 = sess.run([classes, offset, anchors, loss_cls, loss_L1],
                                                  feed_dict={TRAIN_X: train_x,
                                                             TRAIN_ANCHORS: y_anchors,
                                                             TRAIN_CLASSES: y_classes})
        print(np.shape(classes1))
        print(loss_cls1)
        print(loss_L11)












if __name__ == '__main__':
     for i in range(10):
         print(i)