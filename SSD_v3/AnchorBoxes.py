import numpy as np
import keras.backend as K
from ssd_box_encoder import convert_coordinates
class AnchorBoxes(object):
    def __init__(self, img_height, img_width, aspect_ratios, scales):
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_boxes = len(aspect_ratios)

    def __call__(self, x):
        batch_size,  feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
        size = min(self.img_height, self.img_width)
        step = size / min(feature_map_width, feature_map_height)
        wh_list = []
        for ar in self.aspect_ratios:
            box_height = self.img_height * self.scales * np.sqrt(ar)  #返回到原图上对应像素长度
            box_width = self.img_width * self.scales / np.sqrt(ar)
            wh_list.append([box_width, box_height])
        wh_list = np.array(wh_list)
        # 产生anchor中心点的行坐标向量shape=(1, feature_map_height)
        offset_height = 0.5
        offset_width = 0.5
        c_row = np.linspace(start=offset_height * step,
                            stop=(offset_height + feature_map_height - 1) * step,
                            num=feature_map_height)
        # 产生anchor中心点的列坐标向量shape=(1, feature_map_width)
        c_col = np.linspace(start=offset_width * step,
                            stop=(offset_width + feature_map_width - 1) * step,
                            num=feature_map_width)
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
        anchors = convert_coordinates(tensor=anchors, img_width=self.img_width, img_height=self.img_height)
        #转换anchor box的坐标，从中心原点转换到左上角原点
        anchors = K.tile(K.constant(anchors, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))
        return anchors











