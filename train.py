import tensorflow as tf
from models.foot_detector_v1 import FootNet
from roi_generator import pos_neg_roi_generator
from utils import load_data
train_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
test_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
epoch = 10
x = tf.placeholder(tf.float32, [1, 128, 59, 3])
ROI = tf.placeholder(tf.float32, [20, 4])
y = tf.placeholder(tf.float32, [20, 2])
model = FootNet(x, ROI)
cls_label = model.predict()
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=cls_label))
opt1 = tf.train.AdadeltaOptimizer(0.01, rho=0.9)
opt = opt1.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi = load_data(train_path, test_path)
    for i in range(epoch):
        print(i)
        img = train_x[i, :, :, :]
        rois, cls_label = pos_neg_roi_generator(train_roi, 10)
        sess.run(opt, feed_dict={x: img, y: cls_label, ROI: rois})



