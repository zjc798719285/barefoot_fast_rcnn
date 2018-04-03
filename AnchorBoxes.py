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


class AnchorBoxes(object):
    def __init__(self, img_height, img_width, aspect_ratios, scales): #此处输入尺寸为resize后，最短边为600
        self.img_height = img_height
        self.img_width = img_width
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.n_boxes = len(aspect_ratios)

    def get_img_output_length(self, width, height, strides):
        def get_output_length(input_length):
            # zero_pad
            # apply 4 strided convolutions

            for _ in range(strides):
                input_length = int(np.ceil(input_length / 2))
            return input_length

        return get_output_length(width), get_output_length(height)

    def __call__(self, scales):
        anchor_list = []
        feature_map_width, feature_map_height = self.get_img_output_length(self.img_width, self.img_height, scales)
        # feature_map_height = int(self.img_height / scales)
        # feature_map_width = int(self.img_height / scales)
        size = min(self.img_height, self.img_width)
        stride = round(size / min(feature_map_width, feature_map_height)) #stride：feautre_map上移动一个像素
                                                                          #对应于原图多少个像素，即是feature_map相对
                                                                          #于原图缩小的比例
        for aspect_i in self.aspect_ratios:
            for scales_i in self.scales:                   #此处scales_i应该是相对于resize图短边的比例
                box_height = size * scales_i * np.sqrt(aspect_i)  #生成anchor的高度
                box_width = size * scales_i / np.sqrt(aspect_i)   #生成anchor的宽度
                for cx in range(feature_map_width):       #cx：anchor相对于feature_map的中心坐标（row）
                    for cy in range(feature_map_height):    #cy：anchor相对于feature_map的中心坐标（column）
                        xmin = stride * (cy + 0.5) - box_width / 2    #显示器坐标系，左上角为原点，横向x，纵向y
                        xmax = stride * (cy + 0.5) + box_width / 2
                        ymin = stride * (cx + 0.5) - box_height / 2
                        ymax = stride * (cx + 0.5) + box_height / 2
                        anchor_corrd = np.array([xmin, ymin, xmax, ymax])
                        anchor_corner = convert_coordinates(anchor=anchor_corrd,        #转换矩形框的绝对坐标[xmin，ymin，xmax，ymax]
                                                            img_height=self.img_height, #[x,y,w,h]相对坐标
                                                            img_width=self.img_width)
                        anchor_list.append(anchor_corner)
        anchors = np.array(anchor_list)
        # anchors = np.expand_dims(anchors, axis=0)  #给anchor增加batch_size维度
        # anchors = K.tile(K.constant(anchors, dtype='float32'), (1, 1, 1))  #将ndarray覆盖成tensor
        return anchors   #shape=[1, num_boxex, 4], [x, y ,w, h]相对值











