import tensorflow as tf
from models import FootNet_v2
from roi_generator import pos_neg_roi_generator, loss_box, box_roi_generator
from utils import load_data
import numpy as np
train_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
test_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
epoch = 10
x = tf.placeholder(tf.float32, [1, 128, 59, 3])
ROI = tf.placeholder(tf.float32, [20, 4])
gt_roi = tf.placeholder(tf.float32, [20, 4])
y = tf.placeholder(tf.float32, [20, 2])
base_net = FootNet_v2.base_net(x=x, trainable=True)
cls_label = FootNet_v2.classcify(base_net=base_net, rois=ROI, out_size=[4, 2], trainable=True)
box = FootNet_v2.box_regressor(base_net=base_net, rois=ROI, out_size=[4, 2], trainable=True)
cls_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=cls_label))
roi_loss = loss_box(gt=gt_roi, dr=box)
opt1 = tf.train.AdadeltaOptimizer(0.01, rho=0.9)
cls_opt = opt1.minimize(cls_loss)
roi_opt = opt1.minimize(roi_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi = load_data(train_path, test_path)
    img = np.ndarray([1, 128, 59, 3])
    for i in range(epoch):
        print(i)
        img[0, :, :, :] = train_x[i, :, :, :]
        rois, cls_label = pos_neg_roi_generator(train_roi[i], 10)
        gt_box_roi = box_roi_generator(cls_label=cls_label, roi=train_roi[i])
        sess.run(cls_opt, feed_dict={x: img, y: cls_label, ROI: rois})
        sess.run(roi_opt, feed_dict={x: img, gt_roi: gt_box_roi, ROI: rois})



