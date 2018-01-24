import tensorflow as tf
from models import FootNet_v3
import Loss
from roi_generator import pos_neg_roi_generator, box_roi_generator, iou_eval
from utils import load_data
import numpy as np
from config import cfg
import math
def train(img, ground_truth, test_img,test_roi, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    PosNegRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    GroundTruthRoi = tf.placeholder(tf.float32, [params.num_rois, 4])
    GroundRpnRoi = tf.placeholder(tf.float32, [int(params.num_rois/2), 4])
    ClsLabel = tf.placeholder(tf.float32, [params.num_rois, 2])


   # ConvNet define and loss calculation
    base_net = model.base_net(x=Image, trainable=True)
    cls_label = model.classcify(base_net=base_net, rois=PosNegRoi, out_size=params.roi_shape, trainable=True)
    box = model.box_regressor(base_net=base_net, rois=PosNegRoi, out_size=params.roi_shape, trainable=True)
    pre_box = model.box_regressor(base_net=base_net, rois=GroundRpnRoi, out_size=params.roi_shape, trainable=True)
    RPN_rois = model.RPN(base_net=base_net, out_size=params.roi_shape, trainable=True, num_rois=int(params.num_rois/2))
    RPN_loss = Loss.loss_RPN(RPN_rois=RPN_rois, gt=GroundRpnRoi, num_rois=int(params.num_rois/2), mode=params.box_loss)
    cls_loss = Loss.loss_classify(cls_predic=cls_label, labels=ClsLabel)
    roi_loss = Loss.loss_box_regressor(gt=GroundTruthRoi, dr=box, mode=params.box_loss)
    opt1 = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate, rho=params.rho)
    loss = cls_loss + params.loss_balance*roi_loss
    loss_opt = opt1.minimize(loss)
    rpn_opt = opt1.minimize(RPN_loss)
    # training
#    IOU = iou_eval(gt=GroundTruthRoi, dr=box)


    # sess and run,feed data
    with tf.Session() as sess:
        train_x = img; train_roi = ground_truth; test_x = test_img
        sess.run(tf.global_variables_initializer())
        img = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            for step in range(int(math.ceil(len(train_x)/params.batch_shape[0]))):
                img[0, :, :, :] = train_x[step, :, :, :]
                pos_neg_rois, cls_label, rpn_rois = pos_neg_roi_generator(train_roi[step], int(params.num_rois/2))
                gt_box_roi = box_roi_generator(cls_label=cls_label, roi=train_roi[step])
                sess.run(rpn_opt, feed_dict={Image: img, GroundRpnRoi: rpn_rois})
                sess.run(loss_opt, feed_dict={Image: img, GroundTruthRoi: gt_box_roi,
                                               PosNegRoi: pos_neg_rois, ClsLabel: cls_label})
                if step % 50 == 0:
                    print('epoch=', i, 'train_step=', step)
                # testing
                if step % 500 == 0:
                    for step_t in range(int(math.ceil(len(test_x)/params.batch_shape[0]))):
                        img = np.ndarray(params.batch_shape)
                        img[0, :, :, :] = test_x[step_t, :, :, :]
                        pre_rois = sess.run(RPN_rois, feed_dict={Image: img})
                        # pre_box_roi = sess.run(pre_box, feed_dict={Image: img, GroundRpnRoi: pre_rois})
                        if step_t % 50 == 0:
                            print('epoch=', i, 'test_step=', step_t, 'rpn_loss=')



if __name__ =='__main__':

    model = FootNet_v3.FootNet_v3()
    train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
    train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
