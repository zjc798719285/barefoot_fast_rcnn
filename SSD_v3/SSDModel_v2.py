from AnchorBoxes2 import AnchorBoxes
import tensorflow as tf
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
                 n_classes,
                 aspect_ratios,
                 scales):
        self.l2_reg = 0
        self.n_classes = n_classes + 1            # Account for the background class.
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.init = 'glorot_normal'
        self.basenet_trainable = True

    def __call__(self, x):
        net1 = Conv2D(64, (7, 7), padding='same', strides=[2, 2],
                  kernel_initializer=self.init, activation='relu', trainable=self.basenet_trainable)(x)
        net1 = block(x=net1, f=64, init=self.init, trainable=self.basenet_trainable)
        net2 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.basenet_trainable)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net2)
        net2 = block(x=net2, f=128, init=self.init, trainable=self.basenet_trainable)
        net2 = block(x=net2, f=128, init=self.init, trainable=self.basenet_trainable)
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.basenet_trainable)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net3)
        net3 = block(x=net3, f=256, init=self.init, trainable=self.basenet_trainable)
        net3 = block(x=net3, f=256, init=self.init, trainable=self.basenet_trainable)
        net4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net3)
        net4 = Conv2D(512, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=self.init, trainable=self.basenet_trainable)(net4)
        net4 = block(x=net4, f=512, init=self.init, trainable=self.basenet_trainable)
        n_boxes = len(self.aspect_ratios)*len(self.scales)
        # classes2 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
        #                   kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                   name='classes6', activation='sigmoid')(net2)
        classes3 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), padding="same",
                          kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(net3)
        classes3 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), padding="same",
                          kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(classes3)
        classes3 = tf.nn.softmax(classes3)
        classes4 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), padding="same",
                          kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(net4)
        classes4 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), padding="same",
                          kernel_initializer='glorot_uniform', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(classes4)
        classes4 = tf.nn.softmax(classes4)
        # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
        # boxes_offset2 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
        #                 kernel_initializer=self.init, kernel_regularizer=l2(self.l2_reg),
        #                 name='boxes6')(net2)
        boxes_offset3 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
                        kernel_initializer ='zeros', activation='tanh',
                        kernel_regularizer=l2(self.l2_reg), name='boxes3')(net3)
        boxes_offset3 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
                               kernel_initializer='zeros', activation='tanh',
                               kernel_regularizer=l2(self.l2_reg), name='boxes3')(boxes_offset3)
        # boxes_offset3 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
        #                        kernel_initializer='zeros', activation='linear',
        #                        kernel_regularizer=l2(self.l2_reg), name='boxes3')(boxes_offset3)
        boxes_offset4 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
                        kernel_initializer='zeros', kernel_regularizer=l2(self.l2_reg),
                        name='boxes4', activation='tanh')(net4)
        boxes_offset4 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
                               kernel_initializer='zeros', kernel_regularizer=l2(self.l2_reg),
                               name='boxes4', activation='tanh')(boxes_offset4)
        # boxes_offset4 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same",
        #                        kernel_initializer='zeros', kernel_regularizer=l2(self.l2_reg),
        #                        name='boxes4', activation='linear')(boxes_offset4)

        # anchor2 = AnchorBoxes(img_height=128, img_width=59,
        #                       aspect_ratios=self.aspect_ratios,
        #                       scales=self.scales)(boxes_offset2)
        anchor3 = AnchorBoxes(img_height=128, img_width=59,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales)(boxes_offset3)
        anchor4 = AnchorBoxes(img_height=128, img_width=59,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales)(boxes_offset4)
        # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, he
        # We want the four box coordinates isolated in the last axis to compute the smooth
        # classes2_reshaped = Reshape((-1, self.n_classes), name='classes2_reshape')(classes2)
        classes3_reshaped = Reshape((-1, self.n_classes), name='classes3_reshape')(classes3)
        classes4_reshaped = Reshape((-1, self.n_classes), name='classes4_reshape')(classes4)
        # boxes_offset2_reshaped = Reshape((-1, 4), name='boxes2_reshape')(boxes_offset2)
        boxes_offset3_reshaped = Reshape((-1, 4), name='boxes3_reshape')(boxes_offset3)
        boxes_offset4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes_offset4)
        # anchor2_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor2)
        anchor3_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor3)
        anchor4_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor4)
        classes_concat = Concatenate(axis=1, name='classes_concat')([classes3_reshaped,
                                                                     classes4_reshaped])
        # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
        boxes_offset_concat = Concatenate(axis=1, name='boxes_concat')([boxes_offset3_reshaped,
                                                                        boxes_offset4_reshaped])


        anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchor3_reshaped,
                                                                     anchor4_reshaped])

        return classes_concat, boxes_offset_concat, anchors_concat
