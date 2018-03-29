from SSDModel import SSDModel
import tensorflow as tf
import numpy as np
from box_encoder import iou_eval
import cv2
from BatchGenerator import BatchGenerator, load_data
import Loss

######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
save_path= 'E:\PROJECT\\barefoot_fast_rcnn\SSD\AnchorBoxes_unit_test\\'
batch_size = 1
num_boxes_one_image = 7400
pos_neg_ratio = 3
#############
# Load Data #
#############

def draw_rect(image, rect):
    (batch_size, height, width, channel) = np.shape(image)
    for rect_i in rect:
        x = int(height * rect_i[0]); y = int(width * rect_i[1])
        w = int(width * rect_i[2]); h = int(height * rect_i[3])
        image[0, x, y:y + w] = 255
        image[0, x + h, y:y + w] = 255
        image[0, x:x + h, y] = 255
        image[0, x:x + h, y + w] = 255
    return image



def anchor_select(anchors, roi):
    rois = np.array(roi)
    roi_i = rois
    anchor_list = []
    anchor = anchors[0, :, :]
    for anchor_i in anchor:
        if iou_eval(gt=roi_i, dr=anchor_i) > 0.3 and iou_eval(gt=roi_i, dr=anchor_i) <= 0.4:
            anchor_list.append(anchor_i)
    return np.array(anchor_list)

TRAIN_X = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
TRAIN_ANCHORS = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 4])
TRAIN_CLASSES = tf.placeholder(tf.float32, [batch_size, num_boxes_one_image, 2])
classes, offset, anchors = SSDModel(l2_regularization=0,
                                    n_classes=1,
                                    aspect_ratios=[1, 1.5, 2, 2.5, 3, 3.5, 4],
                                    scales=[0.2, 0.3875, 0.575, 0.7625, 0.95],
                                    detect_kernel=(3, 3))(TRAIN_X)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    print(train_cls)
    trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=batch_size)
    testData = BatchGenerator(image=test_x, roi=test_roi, classes=test_cls, batch_size=batch_size)
    for i in range(100):
        train_x, train_roi_list, train_class_list = trainData.next_batch()
        anchors_pred = sess.run(anchors, feed_dict={TRAIN_X: train_x})
        anchors_select = anchor_select(anchors_pred, train_roi_list)
        rect_image = draw_rect(image=train_x, rect=anchors_select)
        path = save_path + str(i) + '.jpg'
        print(train_class_list)

        cv2.imwrite(path, rect_image[0, :, :, :])















if __name__ == '__main__':
     for i in range(10):
         print(i)