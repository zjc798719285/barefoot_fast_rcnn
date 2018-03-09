from SSDModel import SSDModel
import tensorflow as tf
import cv2
import numpy as np
from ssd_box_encoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data

x = tf.placeholder(tf.float32, [32, 866, 399, 3])
classes, offset, anchors = SSDModel(l2_regularization=0, n_classes=1, aspect_ratios=[0.5, 1, 2])(x)

with tf.Session() as sess:
     train_txt = 'F:\zjc\keras_faster_rcnn\data_txt\\train.txt'
     test_txt = 'F:\zjc\keras_faster_rcnn\data_txt\\test.txt'
     train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
     train = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=32)
     image, roi_list, classes_list = train.next_batch()

     # batch_x = np.zeros([1, 866, 389, 3])
     # roi_list = [np.array([[0.14, 0.2, 0.6, 0.7]])]
     # classes_list = [np.array([[1]])]
     # filepath ='E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\V1.0.0.0\\1\\1.jpg'
     # image = cv2.imread(filepath)
     # batch_x[0, :, :, :] = image
     sess.run(tf.global_variables_initializer())
     classes, offset, anchors = sess.run([classes, offset, anchors], feed_dict={x: image})
     y_classes, y_anchors = ssd_box_encoder_batch(roi_list=roi_list,
                                                  classes_list=classes_list,
                                                  anchors=anchors,
                                                  iou_threshold=0.5,
                                                  num_classes=1)
     print(np.shape(y_classes))
     print(np.shape(y_anchors))
     print('classes=', np.sum(np.array(y_classes)))
     print('anchors=', np.sum(np.array(y_anchors)))


# roi_list = [np.array([[0.1, 0.2, 0.1, 0.1]])]
# classes_list = [np.array([[1]])]
# anchors = np.zeros([1, 20646, 4])
# y_classes, y_anchors = ssd_box_encoder_batch(roi_list=roi_list,
#                                              classes_list=classes_list,
#                                              anchors=anchors,
#                                              iou_threshold=0.5,
#                                              num_classes=1)
# print(np.shape(y_classes))
# print(np.shape(y_anchors))
