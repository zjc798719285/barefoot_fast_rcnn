import tensorflow as tf
from models import FootNet_v3
import Loss
from roi_generator import cls_roi_generator, iou_eval
import numpy as np
import math

def train(img, ground_truth, test_img,test_roi, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    ClsRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])   # class proposed rois
    ClsGtRoi = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 4])  #class groundtruth rois
    ClsLabel = tf.placeholder(tf.float32, [int((params.num_cls + 1) * params.num_rois), 2])

   # base net
    base_net = model.base_net(x=Image, trainable=True)
   # Rpn
    rpn_roi_predict = model.RPN(base_net=base_net,
                                out_size=params.roi_shape,
                                trainable=True,
                                num_rois=int((params.num_cls + 1) * params.num_rois))  #一个RPN网络输出多少个推荐框
    rpn_loss = Loss.loss_RPN(RPN_rois=rpn_roi_predict,
                             gt=ClsGtRoi,
                             num_rois=int((params.num_cls + 1) * params.num_rois),
                             mode=params.box_loss)
   # classcify
    cls_predict = model.classcify(base_net=base_net,
                                  rois=ClsRoi,                   #根据ROI预测每一个roi的类别属性
                                  out_size=params.roi_shape,
                                  trainable=True)
    cls_loss = Loss.loss_classify(cls_predic=cls_predict,
                                  labels=ClsLabel)
    #ROI regression
    roi_predict = model.box_regressor(base_net=base_net,
                                      rois=ClsRoi,              #根据ROI预测box，训练时用ClsRoi，测试用RpnRoi
                                      out_size=params.roi_shape,
                                      trainable=True)
    roi_loss = Loss.loss_box_regressor(gt=ClsGtRoi,
                                       dr=roi_predict,
                                       mode=params.box_loss)

    loss = cls_loss + params.loss_balance * roi_loss
    # optimatic
    opt1 = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate, rho=params.rho)
    opt_loss = opt1.minimize(loss)
    opt_rpn = opt1.minimize(rpn_loss)
    # training

    train_x = img; train_roi = ground_truth; test_x = test_img; test_gt_roi = test_roi
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _Image = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            for step in range(int(math.ceil(len(train_x)/params.batch_shape[0]))):
                _Image[0, :, :, :] = train_x[step, :, :, :]
                _ClsRoi, _ClsLabel, _ClsGtRoi = cls_roi_generator(train_roi[step, :], params.num_rois, 1)
                _opt_rpn, _rpn_loss = sess.run([opt_rpn, rpn_loss], feed_dict={Image: _Image,
                                                                               ClsGtRoi: _ClsGtRoi})
                _opt_loss, _cls_loss, _roi_loss = sess.run([opt_loss, cls_loss, roi_loss],
                                       feed_dict={Image: _Image,
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
                        _roi_predict, _cls_predict = sess.run([roi_predict, cls_predict],
                                                      feed_dict={Image: _Image_t,
                                                                 ClsRoi: _rpn_roi_predict})
                        for roi_i in _roi_predict:
                            print(iou_eval(gt=test_gt_roi[step], dr=roi_i))
                       # print(_roi_predict, _cls_predict)



if __name__ =='__main__':

    model = FootNet_v3.FootNet_v3()
    # train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
    # train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
