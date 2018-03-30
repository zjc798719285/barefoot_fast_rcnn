import os
import cv2
import scipy.io as sio
import numpy as np
from box_encoder_decoder import rpn_box_encoder
from AnchorBoxes import AnchorBoxes
class BatchGenerator(object):
    def __init__(self, info_list):
        self.list = info_list
        self.step = 1

    def shuffle(self):
        r = np.random.permutation(len(self.list))
        self.list = self.list[r]
        self.step = 1

    def next_batch(self):
        if self.step > np.floor(len(self.list)):
            self.shuffle()
        list_i = self.list[self.step]
        obj = list_i['obj']; img_width = list_i['width']; img_height = list_i['height'];image_path = list_i['Image']
        img_resize, resize_height, resize_width = get_image(path=image_path, img_height=img_height, img_width=img_width)
        anchors = AnchorBoxes(img_height=resize_height, img_width=resize_width,
                             aspect_ratios=[0.5, 1, 2], scales=[0.2, 0.4, 0.8])(16)  #16：表示basenet输出的feature map与原图缩小比例
        classes, offset, obj_names = rpn_box_encoder(obj=obj, anchors=anchors, iou_pos_thresh=0.5, iou_neg_thresh=0.1)
        #classes: np数组，[1, num_boxes, 2]; offset: np数组，[1, num_boxes, 4]; obj_names: 列表
        self.step += 1
        classes = np.expand_dims(classes, axis=0)      #增加维度，batch_size
        offset = np.expand_dims(offset, axis=0)
        img_resize = np.expand_dims(img_resize, axis=0)
        return img_resize, classes, offset, obj_names, anchors

def get_image(path, img_height, img_width):
    if img_height >= img_width:
        resize_width = 600
        resize_height = int(img_height * (600/img_width))
    else:
        resize_height = 600
        resize_width = int(img_width * (600/img_height))
    img = cv2.imread(path)
    img_resize = cv2.resize(img, (resize_width, resize_height))
    return img_resize, resize_height, resize_width


if __name__ == '__main__':
    # unit testing interference
    train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
    # test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
    # train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
    # print(np.shape(train_x))
    # print(np.shape(train_x))
    # print(np.shape(train_roi))
    # print(np.shape(test_x))
    # print(np.shape(test_roi))
    # print(np.shape(train_cls))
    # print(np.shape(test_cls))
    # print('-------------------------------')
    # train = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=32)
    # image, roi_list, classes_list = train.next_batch()
    # print(np.shape(image))
    # print(np.shape(roi_list))
    # print(np.shape(classes_list))



    # a = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    # b = np.array([1, 2, 3])
    # print(a)
    # print(b)
    # r = np.random.permutation(len(a))
    # a = a[r, :]
    # b = b[r]
    # print(a)
    # print(b)



