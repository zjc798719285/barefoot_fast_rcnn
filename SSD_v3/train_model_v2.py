from SSDModel_v2 import SSDModel
# from resnet import SSDModel
import tensorflow as tf
import numpy as np
from Monitor import Monitor
import scipy.io as sio
from box_filter import box_filter, rect_iou, class_acc
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
import Loss

######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
monitor_path = 'E:\PROJECT\\barefoot_fast_rcnn\SSD_v3\monitor\monitor.mat'
save_path = 'E:\PROJECT\\barefoot_fast_rcnn\SSD_v3\checkpoint\\'
batch_size = 5
num_boxes_one_image = 1920
pos_neg_ratio = 3
test_step = 50
max_iou = 0
#############
# Load Data #
#############

TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 2])
classes, offset, anchors = SSDModel(n_classes=1, aspect_ratios=[2.5, 3, 3.2], scales=[47, 52, 56, 59])(TRAIN_X)
loss_loc, loss_cls, values = Loss.cls_loc_loss(anchor_pred=offset,    #此处函数名称要换
                                               anchor_true=TRAIN_ANCHORS,
                                               y_pred=classes,
                                               y_true=TRAIN_CLASSES,
                                               pos_neg_ratio=pos_neg_ratio)
loss = loss_cls +10*loss_loc                       #一个非常重要的参数，控制分类网络收敛速度
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
opt = optimizer.minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=batch_size)
    testData = BatchGenerator(image=test_x, roi=test_roi, classes=test_cls, batch_size=batch_size)
    train_x, train_roi_list, train_class_list = trainData.next_batch()
    classes2, offset2, anchors2 = sess.run([classes, offset, anchors], feed_dict={TRAIN_X: train_x})
    train_key = ['cls_loss', 'loc_loss', 'anchor_iou', 'rect_iou']
    train_monitor = Monitor(train_key, save_path)
    for i in range(10000):
        print('train_step', i)
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
        print('**')
        acc, recall, num_pos, num_neg, num_hard = class_acc(_cls_pred=cls_pred, _cls_true=y_classes)
        filted_classes, filted_offset, filted_anchors, filted_rect = box_filter(pred_classes=cls_pred,
                                                                                pred_anchors=anchors_pred,
                                                                                pred_offset=offset_pred)
        mean_iou_anchors = rect_iou(roi_list=train_roi_list, rect_batch=filted_anchors)
        mean_iou_rect = rect_iou(roi_list=train_roi_list, rect_batch=filted_rect)
        print('***')
        train_monitor.insert('cls_loss', loss_cls1); train_monitor.insert('loc_loss', loss_cls1)
        # train_monitor.insert('acc', acc); train_monitor.insert('recall', recall)
        train_monitor.insert('anchor_iou', mean_iou_anchors);train_monitor.insert('rect_iou', mean_iou_rect)
        train_monitor.show();train_monitor.save()
        if i % 10 == 0:
            sum_rect_iou = 0
            for j in range(test_step):
                print('test_step=', j)
                test_x, test_roi_list, test_class_list = testData.next_batch()
                cls_pred, offset_pred, anchors_pred= sess.run([classes, offset, anchors],feed_dict={TRAIN_X: test_x})
                acc, recall, num_pos, num_neg, num_hard = class_acc(_cls_pred=cls_pred, _cls_true=y_classes)
                filted_classes, filted_offset, filted_anchors, filted_rect = box_filter(pred_classes=cls_pred,
                                                                                        pred_anchors=anchors_pred,
                                                                                        pred_offset=offset_pred)
                mean_iou_anchors = rect_iou(roi_list=test_roi_list, rect_batch=filted_anchors)
                mean_iou_rect = rect_iou(roi_list=test_roi_list, rect_batch=filted_rect)
                sum_rect_iou += mean_iou_rect
            if max_iou <= sum_rect_iou / test_step:
                max_iou = sum_rect_iou / test_step
                model_name = save_path + str(i) + '.ckpt'
                saver.save(sess, model_name)



                # print('step=', i)
                # print('loss_classes=', loss_cls1, 'loss_L1=', loss_loc1)
                # print('acc=', acc, 'recall=', recall, 'num_pos=', num_pos,
                #      'num_hard=', num_hard, 'num_neg=', num_neg)
                # print('rect_shape', np.shape(filted_rect[1]),
                #     'mean_iou_anchors=', mean_iou_anchors,
                #     'mean_iou_rect=', mean_iou_rect)
                # print('values_shape=', np.shape(values1),
                #     'anchor_shape=', np.shape(filted_anchors[1]))






















if __name__ == '__main__':
     for i in range(10):
         print(i)