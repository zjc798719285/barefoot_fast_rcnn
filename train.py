import tensorflow as tf
from models import FootNet_v4
import Loss
from data_generator import pos_neg_roi_generator, iou_eval, roi_prop_generator
from roi_generator import roi_filter, roi_check
import numpy as np
import math

def train(img, ground_truth, test_img,test_roi, model, params):
    # placeholder define
    Image = tf.placeholder(tf.float32, params.batch_shape)
    ClsRoi = tf.placeholder(tf.float32, [int(params.num_cls*params.num_rois*2), 4])   # class proposed rois
    ClsLabel = tf.placeholder(tf.float32, [int(params.num_cls * params.num_rois*2), 2])
    RegreRoi = tf.placeholder(tf.float32, [int(params.num_cls*params.num_rois), 4])
    RegreGt = tf.placeholder(tf.float32, [int(params.num_cls * params.num_rois), 4])


   # base net
    base_net = model.base_net(x=Image, trainable=True)

   # classcify
    cls_predict = model.classcify(base_net=base_net,
                                  rois=ClsRoi,                   #根据ROI预测每一个roi的类别属性
                                  out_size=params.roi_shape,
                                  trainable=True)
    cls_predict_test = model.classcify(base_net=base_net,
                                  rois=RegreRoi,                   #根据ROI预测每一个roi的类别属性
                                  out_size=params.roi_shape,
                                  trainable=True)
    cls_loss = Loss.loss_classify(cls_predic=cls_predict,
                                  labels=ClsLabel)
    #ROI regression
    roi_shift = model.box_regressor(base_net=base_net,
                                      rois=RegreRoi,              #根据ROI预测box，训练时用ClsRoi，测试用RpnRoi
                                      out_size=params.roi_shape,
                                      trainable=True)
    roi_loss = Loss.loss_box_regressor(gt=RegreGt,
                                       rois=RegreRoi,
                                       dr=roi_shift,
                                       mode=params.box_loss)
    loss = cls_loss + params.loss_balance * roi_loss
    # optimatic
    opt1 = tf.train.AdadeltaOptimizer(learning_rate=params.learning_rate, rho=params.rho)
    opt_loss = opt1.minimize(loss)
    # training
    train_x = img; train_roi = ground_truth; test_x = test_img; test_gt_roi = test_roi
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _Image = np.ndarray(params.batch_shape)
        for i in range(params.epoch):
            for step in range(int(math.ceil(len(train_x)/params.batch_shape[0]))):
                _Image[0, :, :, :] = train_x[step, :, :, :]
                _ClsRoi, _ClsLabel, _RegreRoi, _RegreGt = pos_neg_roi_generator(rois=train_roi[step],
                                                                                cls=[0],
                                                                                num_rois=params.num_rois,
                                                                                num_cls=params.num_cls)

                _opt_loss, _cls_loss, _roi_loss = sess.run([opt_loss, cls_loss, roi_loss],
                                        feed_dict={Image: _Image,
                                                   ClsRoi: _ClsRoi,
                                                   ClsLabel: _ClsLabel,
                                                   RegreRoi: _RegreRoi,
                                                   RegreGt: _RegreGt})
                if step % 50 == 0:
                    print('epoch=', i, 'train_step=', step, 'cls_loss=', _cls_loss, 'roi_loss=', _roi_loss)
                #testing

            SUM_IOU = 0
            for step_t in range(int(math.ceil(len(test_x)/params.batch_shape[0]))):
                        _Image_t = np.ndarray(params.batch_shape)
                        _Image_t[0, :, :, :] = test_x[step_t, :, :, :]
                        _RegreRoi = roi_prop_generator(scale_r=[0.7, 0.8, 0.9, 0.95, 0.98],
                                                       scale_c=[0.7, 0.8, 0.9, 0.95, 0.98],
                                                       num_step=5)
                        num_batch = int(len(_RegreRoi)/params.num_rois)
                        _roi_shift = []; _cls_predict = []
                        for ind in range(num_batch):
                            BatchRegreRoi = _RegreRoi[ind*params.num_rois:(ind+1)*params.num_rois, :]
                            _batch_roi_shift, _batch_cls_predict = sess.run([roi_shift, cls_predict_test],
                                                                             feed_dict={Image: _Image_t,
                                                                                        RegreRoi: BatchRegreRoi})
                            _roi_shift.append(_batch_roi_shift); _cls_predict.append(_batch_cls_predict)
                        _roi_shift = np.reshape(a=_roi_shift, newshape=(-1, 4))
                        _cls_predict = np.reshape(a=_cls_predict, newshape=(-1, 2))
                        final_roi, num_get_rois = roi_filter(rois=(_roi_shift + _RegreRoi[0:600, :]), cls=_cls_predict)
                     #   print('final_roi=', final_roi, 'GroundTruth=', test_gt_roi[step_t])
                        IOU = iou_eval(gt=test_gt_roi[step_t], dr=final_roi)
                        print('epoch=', i, 'test_step=', step_t, 'IOU=', IOU,
                              'num_get_rois=', num_get_rois)
                        SUM_IOU += IOU
            print('avg_IOU=', SUM_IOU/int(math.ceil(len(test_x)/params.batch_shape[0])))
                       # print(_roi_predict, _cls_predict)



if __name__ =='__main__':

    model = FootNet_v4.FootNet_v4()
    # train_x, train_roi, test_x, test_roi = load_data(cfg.train_path, cfg.test_path)
    # train(img=train_x, ground_truth=train_roi, model=model, params=cfg)
