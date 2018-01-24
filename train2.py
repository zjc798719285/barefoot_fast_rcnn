import tensorflow as tf
from models import FootNet_v3
import Loss
from roi_generator import pos_neg_roi_generator, box_roi_generator, cls_roi_generator
from utils import load_data
import numpy as np
from config import cfg
import math
def train(img, ground_truth, test_img,test_roi, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    ClsRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])   # class proposed rois
    ClsGtRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])  #class groundtruth rois
    RpnGtRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])
    ClsRpnRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])
    ClsLabel = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 2])

   # ConvNet define
    base_net = model.base_net(x=Image, trainable=True)
    rpn_roi_predict = model.RPN(base_net=base_net, out_size=params.roi_shape, trainable=True, num_rois=int(params.num_rois / 2))
    cls_predict = model.classcify(base_net=base_net, rois=ClsRoi, out_size=params.roi_shape, trainable=True)
    roi_predict = model.box_regressor(base_net=base_net, rois=ClsRoi, out_size=params.roi_shape, trainable=True)
    roi_predict2 = model.box_regressor(base_net=base_net, rois=ClsRpnRoi, out_size=params.roi_shape, trainable=True)
    # loss function define
    rpn_loss = Loss.loss_RPN(RPN_rois=rpn_roi_predict, gt=RpnGtRoi, num_rois=int(params.num_rois/2), mode=params.box_loss)
    cls_loss = Loss.loss_classify(cls_predic=cls_predict, labels=ClsLabel)
    roi_loss = Loss.loss_box_regressor(gt=ClsGtRoi, dr=roi_predict, mode=params.box_loss)
    loss = cls_loss + params.loss_balance * roi_loss
    # optimatic
    opt1 = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate, rho=params.rho)
    opt_loss = opt1.minimize(loss)
    opt_rpn = opt1.minimize(rpn_loss)
    # training

    train_x = img; train_roi = ground_truth; test_x = test_img; test_roi = test_roi
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _Image = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            for step in range(int(math.ceil(len(train_x)/params.batch_shape[0]))):
                _Image[0, :, :, :] = train_x[step, :, :, :]
               # _ClsRoi, _ClsLabel, _RpnGtRoi = pos_neg_roi_generator(train_roi[step], int(params.num_rois/2))
                _ClsRoi, _ClsLabel, _ClsGtRoi = cls_roi_generator(train_roi[step, :], params.num_rois, 1)
                _RpnGtRoi = _ClsRoi
               # _ClsGtRoi = box_roi_generator(cls_label=_ClsLabel, roi=train_roi[step])
                _opt_rpn, _rpn_loss = sess.run([opt_rpn, rpn_loss], feed_dict={Image: _Image, RpnGtRoi: _RpnGtRoi})
                _opt_loss, _cls_loss, _roi_loss = sess.run(
                     [opt_loss, cls_loss, roi_loss], feed_dict={Image: _Image,
                                                                ClsGtRoi: _ClsGtRoi,
                                                                ClsRoi: _ClsRoi,
                                                                ClsLabel: _ClsLabel})
                if step % 50 == 0:
                     print('epoch=', i, 'train_step=', step, 'rpn_loss=', _rpn_loss,
                                           'cls_loss=', _cls_loss, 'roi_loss=', _roi_loss)
                # testing
                if step % 500 == 0:
                    for step_t in range(int(math.ceil(len(test_x)/params.batch_shape[0]))):
                        _Image_t = np.ndarray(params.batch_shape)
                        _Image_t[0, :, :, :] = test_x[step_t, :, :, :]
                        _rpn_roi_predict = sess.run(rpn_roi_predict, feed_dict={Image: _Image_t})
                        _rpn_roi = np.concatenate((_rpn_roi_predict, _rpn_roi_predict), 0)
                        print(_rpn_roi)
          #              _roi_predict2 = sess.run(roi_predict, feed_dict={Image: _Image_t, ClsRoi: _rpn_roi})
          #               if step_t % 50 == 0:
          #                   print('epoch=', i, 'test_step=', step_t)



if __name__ =='__main__':

    model = FootNet_v3.FootNet_v3()
    train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
    train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
