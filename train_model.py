from models.FootNet_v3 import FootNet_v3 as FootNet
import Loss
from pascal_parser import get_data
from BatchGenerator import BatchGenerator
import tensorflow as tf

INPUT_DIR = 'E:\PROJECT\keras-frcnn\VOCtrainval_11-May-2012\VOC2012'
EPOCHES = 500

IMAGE = tf.placeholder(tf.float32, [1, None, None, 3])
CLASSES = tf.placeholder(tf.float32, [1, None, 2])
OFFSET = tf.placeholder(tf.float32, [1, None, 4])


model = FootNet(aspect_ratio=[0.5, 1, 2], scales=[100, 200, 300])
base_net = model.base_net(x=IMAGE)
classes, offset = model.RPN(base_net=base_net)
loss_rpn_cls = Loss.loss_rpn_cls(y_pred=classes, y_true=CLASSES)
loss_rpn_regress = Loss.loss_rpn_regress(y_pred=offset, y_true=OFFSET)
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
    image, classes, offset, names = train_generator.next_batch()
    _rpn_cls, _rpn_regress, _ = sess.run([loss_rpn_cls, loss_rpn_regress, train_op],
                                         feed_dict={IMAGE: image, CLASSES: classes, OFFSET: offset})
    print('epoch', i, 'rpn_cls=', _rpn_cls, 'rpn_regress=', _rpn_regress)


