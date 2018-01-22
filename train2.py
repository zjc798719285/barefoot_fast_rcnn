import tensorflow as tf
from models import FootNet_v3
from Loss import *
from roi_generator import pos_neg_roi_generator, box_roi_generator
from utils import load_data
import numpy as np
from config import cfg
train_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
test_path = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
epoch = 10

def train(img, ground_truth, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    PosNegRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    GroundTruthRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    ClsLabel = tf.placeholder(tf.float32, [params.num_rois, 2])
   # ConvNet define and loss calculation
    base_net = model.base_net(x=Image, trainable=True)
    cls_label = model.classcify(base_net=base_net, rois=PosNegRoi, out_size=params.roi_shape, trainable=True)
    box = model.box_regressor(base_net=base_net, rois=PosNegRoi, out_size=params.roi_shape, trainable=True)
    cls_loss = loss_classify(cls_predic=cls_label, labels=ClsLabel)
    roi_loss = loss_box_regressor(gt=GroundTruthRoi, dr=box, mode='abs')
    opt1 = tf.train.AdadeltaOptimizer(0.01, rho=0.9)
    cls_opt = opt1.minimize(cls_loss)
    roi_opt = opt1.minimize(roi_loss)
    # sess and run,feed data
    with tf.Session() as sess:
        train_x = img; train_roi = ground_truth
        sess.run(tf.global_variables_initializer())
        img = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            print(i)
            img[0, :, :, :] = train_x[i, :, :, :]
            rois, cls_label = pos_neg_roi_generator(train_roi[i], 10)
            gt_box_roi = box_roi_generator(cls_label=cls_label, roi=train_roi[i])
            sess.run(cls_opt, feed_dict={Image: img, ClsLabel: cls_label, PosNegRoi: rois})
            sess.run(roi_opt, feed_dict={Image: img, GroundTruthRoi: gt_box_roi, PosNegRoi: rois})
if __name__ =='__main__':

    model = FootNet_v3.FootNet_v3()
    train_x, train_roi, test_x, test_roi = load_data(train_path, test_path)
    train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
