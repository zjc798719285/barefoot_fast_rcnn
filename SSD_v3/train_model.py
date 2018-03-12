from SSDModel_v2 import SSDModel
import tensorflow as tf
import numpy as np
from box_filter import class_pred_acc, box_filter, batch_mean_iou,class_pred_acc2
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
import Loss

######################
# Parameters setting #
######################
train_txt = 'F:\zjc\data\RCNN\V1.0.0.0_128\\train.txt'
test_txt = 'F:\zjc\data\RCNN\V1.0.0.0_128\\train.txt'
batch_size = 32
num_boxes_one_image = 1248
pos_neg_ratio = 1
#############
# Load Data #
#############

TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 2])
classes, offset, anchors = SSDModel(l2_regularization=0,
                                    n_classes=1,
                                    aspect_ratios=[2.5, 3, 3.2],
                                    scales=[47, 52, 56, 59])(TRAIN_X)



loss_loc, loss_cls = Loss.cls_loc_loss(anchor_pred=anchors, anchor_true=TRAIN_ANCHORS,
                    y_pred=classes, y_true=TRAIN_CLASSES,
                    pos_neg_ratio=pos_neg_ratio)
loss = loss_cls + loss_loc
optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.9)
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
                                                     iou_thresh_pos=0.4,
                                                     iou_thresh_neg=0.1,
                                                     num_classes=1)
        cls_pred, offset_pred, anchors_pred,\
        loss_cls1, loss_loc1, opt1 = sess.run([classes, offset, anchors,
                                               loss_cls, loss_loc, opt],
                                               feed_dict={TRAIN_X: train_x,
                                                          TRAIN_ANCHORS: y_anchors,
                                                          TRAIN_CLASSES: y_classes})
        if i % 5 == 0:
           acc, recall, num_pos, num_hard, num_neg = class_pred_acc2(cls_pred=cls_pred, cls_true=y_classes)
           # pred_rect, pred_anchors, pred_offset = box_filter(pred_offset=offset_pred,
           #                        pred_anchors=anchors_pred,
           #                        pred_classes=cls_pred)
           # mean_iou = batch_mean_iou(roi_list=train_roi_list, rect=pred_rect)

           print('step=', i)
           print('loss_classes=', loss_cls1, 'loss_L1=', loss_loc1, 'rect_shape=')
           print('acc=', acc, 'recall=', recall, 'num_pos=', num_pos,
                 'num_hard=', num_hard, 'num_neg=', num_neg)
           # print('pred_cls=', cls_pred)

















if __name__ == '__main__':
     for i in range(10):
         print(i)