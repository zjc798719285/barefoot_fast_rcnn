import numpy as np
import keras.backend as K


def convert_coordinates(anchor, img_height, img_width):
    xmin = anchor[0]; ymin=anchor[1]; xmax = anchor[2]; ymax=anchor[3]
    x = max(xmin, 0); y = max(ymin, 0)        #保证生成的anchor坐标必须大于0
    h = ymax - y; w = xmax - x
    x = x / img_width; y = y / img_height
    w = w / img_width; h = h / img_height
    anchor_corner = np.array([x, y, w, h])
    return anchor_corner


def convert_coordinates_fast(anchor, img_height, img_width):
    xmin = anchor[:, 0]; ymin=anchor[:, 1]; xmax = anchor[:, 2]; ymax=anchor[:, 3]
    x = np.maximum(xmin, 0); y = np.maximum(ymin, 0)        #保证生成的anchor坐标必须大于0
    h = ymax - y; w = xmax - x
    x = np.reshape(x / img_width, newshape=(-1, 1))
    y = np.reshape(y / img_height, newshape=(-1, 1))
    w = np.reshape(w / img_width, newshape=(-1, 1))
    h = np.reshape(h / img_height, newshape=(-1, 1))
    anchor_corner = np.concatenate((x, y, w, h), axis=-1)
    return anchor_corner

class AnchorBoxes(object):
    def __init__(self, img_height, img_width, aspect_ratios, scales): #此处输入尺寸为resize后，最短边为600
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_boxes = len(aspect_ratios)

    def get_img_output_length(self, width, height):
        def get_output_length(input_length):
            # zero_pad
            # apply 4 strided convolutions
            strides = [2, 2, 2, 2]
            for stride_i in strides:
                input_length = int(np.ceil(input_length / stride_i))
            return input_length

        return get_output_length(width), get_output_length(height)

    def __call__(self, scales):
        anchor_list = []
        feature_map_width, feature_map_height = self.get_img_output_length(self.img_width, self.img_height)
        # feature_map_height = int(self.img_height / scales)
        # feature_map_width = int(self.img_height / scales)
        size = min(self.img_height, self.img_width)
        stride = round(size / min(feature_map_width, feature_map_height)) #stride：feautre_map上移动一个像素
                                                                          #对应于原图多少个像素，即是feature_map相对
                                                                          #于原图缩小的比例
        wh_list = []
        for aspect_i in self.aspect_ratios:
            for scales_i in self.scales:  # 此处scales_i应该是相对于resize图短边的比例
                box_height = size * scales_i * np.sqrt(aspect_i)  # 生成anchor的高度
                box_width = size * scales_i / np.sqrt(aspect_i)  # 生成anchor的宽度
                wh_list.append([box_height, box_width])
        cx = np.linspace(start=0, stop=feature_map_width, num=feature_map_width)
        cy = np.linspace(start=0, stop=feature_map_height, num=feature_map_height)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        anchor_corrd = []
        anchor_i = np.zeros([feature_map_height, feature_map_width, 4])
        for wh_i in wh_list:
            box_height = wh_i[0]; box_width = wh_i[1]
            anchor_i[:, :, 0] = stride * (cx_grid + 0.5) - box_width / 2    #  xmin
            anchor_i[:, :, 1] = stride * (cy_grid + 0.5) - box_height / 2   #  ymin
            anchor_i[:, :, 2] = stride * (cx_grid + 0.5) + box_width / 2
            anchor_i[:, :, 3] = stride * (cy_grid + 0.5) + box_height / 2
            anchor_resh = np.reshape(anchor_i, (-1, 4))
            anchor_corrd.append(anchor_resh)
        anchor_corrd = np.concatenate((anchor_corrd), axis=0)
        anchor_corner = convert_coordinates_fast(anchor=anchor_corrd,
                                                 img_height=self.img_height,
                                                 img_width=self.img_width)
        return anchor_corner











