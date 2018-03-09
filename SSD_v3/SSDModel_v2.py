from AnchorBoxes2 import AnchorBoxes
from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Concatenate, SeparableConv2D
from keras.regularizers import l2


def block(x, f, init, trainable=True):
    b1 = SeparableConv2D(filters=f, kernel_size=(3, 3), padding='same', depth_multiplier=1)(x)
    b1 = Conv2D(f, (1, 1), padding='same', strides=[1, 1], activation='relu',
                kernel_initializer=init, trainable=trainable)(b1)
    b2 = SeparableConv2D(filters=f, kernel_size=(3, 3), padding='same', depth_multiplier=1)(b1)
    b2 = Conv2D(f, (1, 1), padding='same', strides=[1, 1], activation='relu',
                kernel_initializer=init, trainable=trainable)(b2)
    b3 = b2 + x
    return b3

class SSDModel(object):
    def __init__(self,
                 l2_regularization,
                 n_classes,
                 aspect_ratios,
                 scales):
        self.l2_reg = l2_regularization
        self.n_classes = n_classes + 1            # Account for the background class.
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.init = 'glorot_normal'


    def __call__(self, x):
        net1 = Conv2D(64, (7, 7), padding='same', strides=[2, 2],
                  kernel_initializer=self.init, activation='relu')(x)
        net1 = block(x=net1, f=64, init=self.init)
        net2 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2))(net2)
        net2 = block(x=net2, f=128, init=self.init)
        net2 = block(x=net2, f=128, init=self.init)
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2))(net3)
        net3 = block(x=net3, f=256, init=self.init)
        net3 = block(x=net3, f=256, init=self.init)
        net4 = MaxPooling2D((3, 3), strides=(2, 2))(net3)
        net4 = Conv2D(512, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init)(net4)
        net4 = block(x=net4, f=512, init=self.init)
        n_boxes = len(self.aspect_ratios)*len(self.scales)
        # classes4 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
        #                   kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                   name='classes4', activation='sigmoid')(conv4)
        # classes5 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
        #                   kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                   name='classes5', activation='sigmoid')(conv5)
        # classes6 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
        #                   kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                   name='classes6', activation='sigmoid')(conv6)
        classes7 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
                          kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(net3)
        classes8 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
                          kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(net4)

        # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
        # boxes_offset4 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
        #                 kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                 name='boxes4')(conv4)
        # boxes_offset5 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
        #                 kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                 name='boxes5')(conv5)
        # boxes_offset6 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
        #                 kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                 name='boxes6')(conv6)
        boxes_offset7 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
                        kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
                        name='boxes7')(net3)
        boxes_offset8 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes7')(net4)

        # anchor4 = AnchorBoxes(img_height=866, img_width=389,
        #                       aspect_ratios=self.aspect_ratios,
        #                       scales=self.scales[0])(boxes_offset4)
        # anchor5 = AnchorBoxes(img_height=866, img_width=389,
        #                       aspect_ratios=self.aspect_ratios,
        #                       scales=self.scales[1])(boxes_offset5)
        # anchor6 = AnchorBoxes(img_height=866, img_width=389,
        #                       aspect_ratios=self.aspect_ratios,
        #                       scales=self.scales[2])(boxes_offset6)
        anchor7 = AnchorBoxes(img_height=128, img_width=59,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales)(boxes_offset7)
        anchor8 = AnchorBoxes(img_height=128, img_width=59,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales)(boxes_offset8)

        # classes4_reshaped = Reshape((-1, self.n_classes), name='classes4_reshape')(classes4)
        # classes5_reshaped = Reshape((-1, self.n_classes), name='classes5_reshape')(classes5)
        # classes6_reshaped = Reshape((-1, self.n_classes), name='classes6_reshape')(classes6)
        classes7_reshaped = Reshape((-1, self.n_classes), name='classes7_reshape')(classes7)
        classes8_reshaped = Reshape((-1, self.n_classes), name='classes7_reshape')(classes8)
        # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, he
        # We want the four box coordinates isolated in the last axis to compute the smooth
        # boxes_offset4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes_offset4)
        # boxes_offset5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes_offset5)
        # boxes_offset6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes_offset6)
        boxes_offset7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes_offset7)
        boxes_offset8_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes_offset8)
        # anchor4_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor4)
        # anchor5_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor5)
        # anchor6_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor6)
        anchor7_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor7)
        anchor8_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor8)
        classes_concat = Concatenate(axis=1, name='classes_concat')([classes7_reshaped,
                                                                     classes8_reshaped])

        # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
        boxes_offset_concat = Concatenate(axis=1, name='boxes_concat')([boxes_offset7_reshaped,
                                                                        boxes_offset8_reshaped])


        anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchor7_reshaped,
                                                                     anchor8_reshaped])

        return classes_concat, boxes_offset_concat, anchors_concat
