import numpy as np
import keras.backend as K
from keras.engine.topology import Layer

def convert_coordinates(anchor, img_height, img_width):
    xmin = anchor[0]; ymin=anchor[1]; xmax = anchor[2]; ymax=anchor[3]
    x = max(xmin, 0); y = max(ymin, 0)        #保证生成的anchor坐标必须大于0
    h = xmax - x; w = ymax - y
    x = x / img_height; y = y / img_width
    w = w / img_width; h = h / img_height
    anchor_corner = np.array([x, y, w, h])
    return anchor_corner

class AnchorBoxes(object):
    def __init__(self, img_height, img_width, aspect_ratios, scales):
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_boxes = len(aspect_ratios)

    def __call__(self, x):
        anchor_list = []
        batch_size,  feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        size = min(self.img_height, self.img_width)
        stride = round(size / min(feature_map_width, feature_map_height)) #stride：feautre_map上移动一个像素
                                                                          #对应于原图多少个像素，即是feature_map相对
                                                                          #于原图缩小的比例
        for aspect_i in self.aspect_ratios:
            for scales_i in self.scales:                   #此处scales_i应该是在原图上anchor的大小
                box_height = scales_i * np.sqrt(aspect_i)  #生成anchor的高度
                box_width = scales_i / np.sqrt(aspect_i)   #生成anchor的宽度
                for cx in range(feature_map_height):       #cx：anchor相对于feature_map的中心坐标（row）
                    for cy in range(feature_map_width):    #cy：anchor相对于feature_map的中心坐标（column）
                        xmin = stride * (cx + 0.5) - box_height / 2
                        xmax = stride * (cx + 0.5) + box_height / 2
                        ymin = stride * (cy + 0.5) - box_width / 2
                        ymax = stride * (cy + 0.5) + box_width / 2
                        anchor_corrd = np.array([xmin, ymin, xmax, ymax])
                        anchor_corner = convert_coordinates(anchor=anchor_corrd,        #转换矩形框的绝对坐标[xmin，ymin，xmax，ymax]
                                                            img_height=self.img_height, #[x,y,w,h]相对坐标
                                                            img_width=self.img_width)
                        anchor_list.append(anchor_corner)
        anchors = np.array(anchor_list); anchors = np.expand_dims(anchors, axis=0)  #给anchor增加batch_size维度
        anchors = K.tile(K.constant(anchors, dtype='float32'), (K.shape(x)[0], 1, 1))  #将ndarray覆盖成tensor
        return anchors











