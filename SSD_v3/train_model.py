from SSDModel_v2 import SSDModel
# from resnet import SSDModel
import tensorflow as tf
import numpy as np
import scipy.io as sio
from box_filter import box_filter, rect_iou, class_acc
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
import Loss, time
######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
monitor_path = 'E:\PROJECT\\barefoot_fast_rcnn\SSD_v3\monitor\monitor.mat'
batch_size = 5
num_boxes_one_image = 1920
pos_neg_ratio = 5
#############
# Load Data #
#############

TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 2])
classes, offset, anchors = SSDModel(n_classes=1,
                                    aspect_ratios=[2.5, 3, 3.2],
                                    scales=[47, 52, 56, 59])(TRAIN_X)
loss_loc, loss_cls, values = Loss.cls_loc_loss(anchor_pred=offset,    #此处函数名称要换
                                       anchor_true=TRAIN_ANCHORS,
                                       y_pred=classes,
                                       y_true=TRAIN_CLASSES,
                                       pos_neg_ratio=pos_neg_ratio)
loss = loss_cls + loss_loc
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
opt = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=batch_size)
    testData = BatchGenerator(image=test_x, roi=test_roi, classes=test_cls, batch_size=batch_size)
    train_x, train_roi_list, train_class_list = trainData.next_batch()
    classes2, offset2, anchors2 = sess.run([classes, offset, anchors], feed_dict={TRAIN_X: train_x})
    monitor = {'pos_acc': [], 'neg_acc': [], 'cls_loss': [], 'loc_loss': [],
               'anchor_iou': [], 'rect_iou': []}
    for i in range(10000):
        # t1 = time.time()
        train_x, train_roi_list, train_class_list = trainData.next_batch()
        y_classes, y_anchors = ssd_box_encoder_batch(roi_list=train_roi_list,
                                                     classes_list=train_class_list,
                                                     anchors=anchors2,
                                                     iou_thresh_pos=0.5,
                                                     iou_thresh_neg=0.1,
                                                     num_classes=1)
        cls_pred, offset_pred, anchors_pred,\
        loss_cls1, loss_loc1, opt1, values1 = sess.run([classes, offset, anchors,
                                               loss_cls, loss_loc, opt, values],
                                               feed_dict={TRAIN_X: train_x,
                                                          TRAIN_ANCHORS: y_anchors,
                                                          TRAIN_CLASSES: y_classes})
        # t2 = time.time()
        # print('train time:', t2 - t1)
        if i % 5 == 0:
                acc, recall, num_pos, num_neg, num_hard = class_acc(_cls_pred=cls_pred, _cls_true=y_classes)
                # t3 = time.time()
                # print('class_acc time:', t3 - t2)
                filted_classes, filted_offset, filted_anchors, filted_rect = box_filter(pred_classes=cls_pred,
                                                                      pred_anchors=anchors_pred,
                                                                       pred_offset=offset_pred)
                # t4 = time.time()
                # print('box_filter time:', t4 - t3)
                mean_iou_anchors = rect_iou(roi_list=train_roi_list, rect_batch=filted_anchors)
                mean_iou_rect = rect_iou(roi_list=train_roi_list, rect_batch=filted_rect)
                monitor['pos_acc'].append(acc[0]);monitor['cls_loss'].append(loss_cls1)
                monitor['neg_acc'].append(acc[1]);monitor['loc_loss'].append(loss_loc1)
                monitor['anchor_iou'].append(mean_iou_anchors)
                monitor['rect_iou'].append(mean_iou_rect)
                sio.savemat(monitor_path, monitor)

                print('step=', i)
                print('loss_classes=', loss_cls1, 'loss_L1=', loss_loc1)
                print('acc=', acc, 'recall=', recall, 'num_pos=', num_pos,
                     'num_hard=', num_hard, 'num_neg=', num_neg)
                print('rect_shape', np.shape(filted_rect[1]),
                    'mean_iou_anchors=', mean_iou_anchors,
                    'mean_iou_rect=', mean_iou_rect)
                print('values_shape=', np.shape(values1),
                    'anchor_shape=', np.shape(filted_anchors[1]))






















if __name__ == '__main__':
     for i in range(10):
         print(i)