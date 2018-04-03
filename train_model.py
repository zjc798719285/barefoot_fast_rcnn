from models.FootNet_v3 import FootNet_v3 as FootNet
import Loss
from pascal_parser import get_data
from box_encoder_decoder import roi_box_decoder
from BatchGenerator import BatchGenerator
import tensorflow as tf
from RoiPooling import RoiPooling
import time
INPUT_DIR = 'I:\zjc\data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
EPOCHES = 500
NUM_CLASSES = 20


IMAGE = tf.placeholder(tf.float32, [1, None, None, 3])
CLASSES = tf.placeholder(tf.float32, [1, None, NUM_CLASSES + 1])
OFFSET = tf.placeholder(tf.float32, [1, None, 4])
ROIS = tf.placeholder(tf.float32, [None, 4])

model = FootNet(aspect_ratio=[0.5, 1, 2], scales=[100, 200, 300])
base_net = model.base_net(x=IMAGE)
classes_rpn, offset_rpn = model.RPN(base_net=base_net)
loss_rpn_cls = Loss.loss_rpn_cls(y_pred=classes_rpn, y_true=CLASSES)
loss_rpn_regress = Loss.loss_rpn_regress(y_pred=offset_rpn, y_true=OFFSET)
loss = loss_rpn_regress + loss_rpn_cls


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
var_list = tf.trainable_variables()
gradients = optimizer.compute_gradients(loss=loss, var_list=var_list)
train_op = optimizer.apply_gradients(grads_and_vars=gradients)

train_list, test_list = get_data(input_dir=INPUT_DIR)
train_generator = BatchGenerator(info_list=train_list)
test_generator = BatchGenerator(info_list=test_list)
sess = tf.InteractiveSession()
for i in range(EPOCHES):
        sess.run(tf.global_variables_initializer())
     # try:
        t1 = time.time()
        image_x, classes_y, offset_y, obj_names, anchors = train_generator.next_batch()
        t2 = time.time()
        pred_classes, pred_offset, _rpn_cls, _rpn_regress, _ = \
                                    sess.run([classes_rpn, offset_rpn, loss_rpn_cls, loss_rpn_regress, train_op],
                                             feed_dict={IMAGE: image_x, CLASSES: classes_y, OFFSET: offset_y})
        t3 = time.time()
        rect_list, prob_list, name_list = roi_box_decoder(anchors=anchors, classes=pred_classes,
                                                          offset=pred_offset, names=obj_names)
        t4 = time.time()
        print('next_batch:', t2-t1, 'sess:', t3-t2, 'roi:', t4 - t3)
        print('epoch', i, 'rpn_cls=', _rpn_cls, 'rpn_regress=', _rpn_regress, 'num_rect=', len(rect_list))

     # except:
     #      continue





