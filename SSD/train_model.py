from SSDModel import SSDModel
import tensorflow as tf
import numpy as np
from box_filter import class_pred_acc
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
import Loss

######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
batch_size = 32
num_boxes_one_image = 408*3
pos_neg_ratio = 1
#############
# Load Data #
#############

TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 2])
classes, offset, anchors = SSDModel(l2_regularization=0,
                                    n_classes=1,
                                    aspect_ratios=[0.5, 1, 2],
                                    scales=[1, 2, 3])(TRAIN_X)

# loss_cls = Loss.cls_loss(y_pred=classes, y_true=TRAIN_CLASSES)
# loss_L1 = Loss.smooth_L1(anchor_pred=offset, anchor_true=TRAIN_ANCHORS)
# loss = loss_cls + loss_L1

loss_loc, loss_cls= Loss.cls_loc_loss(anchor_pred=anchors, anchor_true=TRAIN_ANCHORS,
                    y_pred=classes, y_true=TRAIN_CLASSES,
                    pos_neg_ratio=pos_neg_ratio)
loss = loss_cls + loss_loc
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1, rho=0.9)
opt = optimizer.minimize(loss)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=batch_size)
    testData = BatchGenerator(image=test_x, roi=test_roi, classes=test_cls, batch_size=batch_size)
    train_x, train_roi_list, train_class_list = trainData.next_batch()
    classes2, offset2, anchors2 = sess.run([classes, offset, anchors], feed_dict={TRAIN_X: train_x})
    for i in range(10000):
        train_x, train_roi_list, train_class_list = trainData.next_batch()
        y_classes, y_anchors = ssd_box_encoder_batch(roi_list=train_roi_list,
                                                     classes_list=train_class_list,
                                                     anchors=anchors2,
                                                     iou_threshold=0.5,
                                                     num_classes=1)
        cls_pred, offset_pred, anchors_pred, loss_cls1, loss_loc1, opt1 = sess.run([classes,
                                                                     offset,
                                                                     anchors,
                                                                     loss_cls,
                                                                     loss_loc,
                                                                     opt],
                                                  feed_dict={TRAIN_X: train_x,
                                                             TRAIN_ANCHORS: y_anchors,
                                                             TRAIN_CLASSES: y_classes})
        if i % 20 == 0:
           acc_bk, acc_cls, num_class_pred, num_class_true = class_pred_acc(cls_pred=cls_pred, cls_true=y_classes)
           print('step=', i)
           print('loss_classes=', loss_cls1, 'loss_L1=', loss_loc1)
           print('acc_bk=', acc_bk, 'acc_cls=', acc_cls)
           print('num_class=', np.sum(y_classes[:, :, 1]), 'num_bk=', np.sum(y_classes[:, :, 0]),
                 'num_class_pred=', num_class_pred, 'num_class_true=', num_class_true)












if __name__ == '__main__':
     for i in range(10):
         print(i)