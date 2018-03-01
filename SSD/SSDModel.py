from AnchorBoxes import AnchorBoxes
from keras.layers import Conv2D, Activation, MaxPooling2D, Reshape, Concatenate
from keras.regularizers import l2

class SSDModel(object):
    def __init__(self,
                 l2_regularization,
                 n_classes,
                 aspect_ratios,
                 scales,
                 detect_kernel):
        self.l2_reg = l2_regularization
        self.n_classes = n_classes + 1            # Account for the background class.
        self.aspect_ratios = aspect_ratios
        self.scales = scales
        self.detect_kernel = detect_kernel
    def __call__(self, x):

        conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv1')(x)
        conv1 = Activation(activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
        conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv2')(pool1)
        conv2 = Activation(activation='relu')(conv2)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv3')(conv2)
        conv3 = Activation(activation='relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
        conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv4')(pool3)
        conv4 = Activation(activation='relu')(conv4)
        conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv5')(conv4)
        conv5 = Activation(activation='relu')(conv5)
        conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv6')(conv5)
        conv6 = Activation(activation='relu')(conv6)
        pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)
        conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv7')(pool6)
        conv7 = Activation(activation='relu')(conv7)
        pool7 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv7)
        conv8 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal',
                       kernel_regularizer=l2(self.l2_reg), name='conv7')(pool7)
        conv8 = Activation(activation='relu')(conv8)
        n_boxes = len(self.aspect_ratios)
        classes4 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                          name='classes4', activation='sigmoid')(conv4)
        classes5 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                          name='classes5', activation='sigmoid')(conv5)
        classes6 = Conv2D(n_boxes * self.n_classes, (3, 3), strides=(1, 1), padding="same",
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                          name='classes6', activation='sigmoid')(conv6)
        classes7 = Conv2D(n_boxes * self.n_classes, self.detect_kernel, strides=(1, 1), padding="same",
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(conv7)
        classes8 = Conv2D(n_boxes * self.n_classes, self.detect_kernel, strides=(1, 1), padding="same",
                          kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                          name='classes7', activation='sigmoid')(conv8)

        # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
        boxes_offset4 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes4', activation='sigmoid')(conv4)
        boxes_offset5 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes5', activation='sigmoid')(conv5)
        boxes_offset6 = Conv2D(n_boxes * 4, (3, 3), strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes6', activation='sigmoid')(conv6)
        boxes_offset7 = Conv2D(n_boxes * 4, self.detect_kernel, strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes7', activation='sigmoid')(conv7)
        boxes_offset8 = Conv2D(n_boxes * 4, self.detect_kernel, strides=(1, 1), padding="same",
                        kernel_initializer='he_normal', kernel_regularizer=l2(self.l2_reg),
                        name='boxes7', activation='sigmoid')(conv8)

        anchor4 = AnchorBoxes(img_height=866, img_width=389,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales[0])(boxes_offset4)
        anchor5 = AnchorBoxes(img_height=866, img_width=389,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales[1])(boxes_offset5)
        anchor6 = AnchorBoxes(img_height=866, img_width=389,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales[2])(boxes_offset6)
        anchor7 = AnchorBoxes(img_height=866, img_width=389,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales[3])(boxes_offset7)
        anchor8 = AnchorBoxes(img_height=866, img_width=389,
                              aspect_ratios=self.aspect_ratios,
                              scales=self.scales[4])(boxes_offset8)

        classes4_reshaped = Reshape((-1, self.n_classes), name='classes4_reshape')(classes4)
        classes5_reshaped = Reshape((-1, self.n_classes), name='classes5_reshape')(classes5)
        classes6_reshaped = Reshape((-1, self.n_classes), name='classes6_reshape')(classes6)
        classes7_reshaped = Reshape((-1, self.n_classes), name='classes7_reshape')(classes7)
        classes8_reshaped = Reshape((-1, self.n_classes), name='classes7_reshape')(classes8)
        # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, he
        # We want the four box coordinates isolated in the last axis to compute the smooth
        boxes_offset4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes_offset4)
        boxes_offset5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes_offset5)
        boxes_offset6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes_offset6)
        boxes_offset7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes_offset7)
        boxes_offset8_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes_offset8)
        anchor4_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor4)
        anchor5_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor5)
        anchor6_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor6)
        anchor7_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor7)
        anchor8_reshaped = Reshape((-1, 4), name='anchor4_reshaped')(anchor8)
        classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                     classes5_reshaped,
                                                                     classes6_reshaped,
                                                                     classes7_reshaped,
                                                                     classes8_reshaped])

        # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
        boxes_offset_concat = Concatenate(axis=1, name='boxes_concat')([boxes_offset4_reshaped,
                                                                        boxes_offset5_reshaped,
                                                                        boxes_offset6_reshaped,
                                                                        boxes_offset7_reshaped,
                                                                        boxes_offset8_reshaped])


        anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchor4_reshaped,
                                                                     anchor5_reshaped,
                                                                     anchor6_reshaped,
                                                                     anchor7_reshaped,
                                                                     anchor8_reshaped])

        return classes_concat, boxes_offset_concat, anchors_concat
