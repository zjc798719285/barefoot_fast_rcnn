import numpy as np
import keras.backend as K
from ssd_box_encoder import convert_coordinates
class AnchorBoxes(object):
    def __init__(self, img_height, img_width, aspect_ratios, scales):
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_boxes = len(aspect_ratios) * len(scales)

    def __call__(self, x):
        batch_size,  feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        wh_list = []
        for ar in self.aspect_ratios:
            box_height = 3 / feature_map_height / np.sqrt(ar)  #默认卷积核3*3，此处可以不写死
            box_width = 3 / feature_map_width * np.sqrt(ar)
            for scales_i in self.scales:
              box_height = box_height * np.sqrt(scales_i)
              box_width = box_width * np.sqrt(scales_i)
              wh_list.append([box_width, box_height])            #矩形框长和宽组成列表
        wh_list = np.array(wh_list)
        # 产生anchor中心点的行坐标向量shape=(1, feature_map_height)
        c_row = np.linspace(start=0, stop=1, num=feature_map_height)
        # 产生anchor中心点的列坐标向量shape=(1, feature_map_width)
        c_col = np.linspace(start=0, stop=1, num=feature_map_width)
        #通过向量坐标生成网格数据，
        #c_row_grid的每一列是c_row的列向量，一共有len(c_col)个c_row列向量组成
        #c_col_grid的每一行是c_col的行向量，一共有len(c_row)个c_col行向量组成
        c_row_grid, c_col_grid = np.meshgrid(c_col, c_row)
        c_row_grid = np.expand_dims(c_row_grid, -1)  # This is necessary for np.tile() to do what we want further down
        c_col_grid = np.expand_dims(c_col_grid, -1)  #
        anchors = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))
        anchors[:, :, :, 0] = np.tile(c_row_grid, (1, 1, self.n_boxes))
        anchors[:, :, :, 1] = np.tile(c_col_grid, (1, 1, self.n_boxes))
        anchors[:, :, :, 2] = wh_list[:, 0]
        anchors[:, :, :, 3] = wh_list[:, 1]
        anchors = np.expand_dims(anchors, axis=0)
        anchors = convert_coordinates(anchors)  #转换anchor box的坐标，从中心原点转换到左上角原点
        anchors = K.tile(K.constant(anchors, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
        return anchors











