import tensorflow as tf
from models import FootNet_v3
from Loss import *
from roi_generator import pos_neg_roi_generator, box_roi_generator, iou_eval
from utils import load_data
import numpy as np
from config import cfg
import math
def train(img, ground_truth,test_img, test_gt, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    ClsRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    ClsGtRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    ClsLabel = tf.placeholder(tf.float32, [params.num_rois, 2])

   # ConvNet define and loss calculation
    base_net = model.base_net(x=Image, trainable=True)
    cls_label = model.classcify(base_net=base_net, rois=ClsRoi, out_size=params.roi_shape, trainable=True)
    box = model.box_regressor(base_net=base_net, rois=ClsRoi, out_size=params.roi_shape, trainable=True)
    cls_loss = loss_classify(cls_predic=cls_label, labels=ClsLabel)
    roi_loss = loss_box_regressor(gt=ClsGtRoi, dr=box, mode=params.box_loss)
    opt1 = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate, rho=params.rho)
    loss = cls_loss + params.loss_balance*roi_loss
    loss_opt = opt1.minimize(loss)

    # testing
#    IOU = iou_eval(gt=GroundTruthRoi, dr=box)



    # sess and run,feed data
    with tf.Session() as sess:
        train_x = img; train_roi = ground_truth
        sess.run(tf.global_variables_initializer())
        img = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            for step in range(int(math.ceil(len(train_x)/params.batch_shape[0]))):
                print('epoch=', i, 'step=', step)
                img[0, :, :, :] = train_x[step, :, :, :]
                rois, cls_label = pos_neg_roi_generator(train_roi[step], int(params.num_rois/2))
                gt_box_roi = box_roi_generator(cls_label=cls_label, roi=train_roi[step])
                sess.run(loss_opt, feed_dict={Image: img, ClsGtRoi: gt_box_roi,
                                              ClsRoi: rois, ClsLabel: cls_label})
                # if step%500 == 0:
                #    iou = sess.run(iou, feed_dict={Image: img, GroundTruthRoi: gt_box_roi,
                #                                   PosNegRoi: rois, ClsLabel: cls_label})

if __name__ =='__main__':

    model = FootNet_v3.FootNet_v3()
    train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
    train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
