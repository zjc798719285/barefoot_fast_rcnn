# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''

from __future__ import print_function
from __future__ import absolute_import
from FixedBatchNormalization import FixedBatchNormalization
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed, Conv2D, Reshape, Concatenate
from AnchorBoxes2 import AnchorBoxes
from keras import backend as K

def get_weight_path():
    if K.image_dim_ordering() == 'th':
        return 'resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
    else:
        return 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

def get_img_output_length(width, height):
    def get_output_length(input_length):
        # zero_pad
        input_length += 6
        # apply 4 strided convolutions
        filter_sizes = [7, 3, 1, 1]
        stride = 2
        for filter_size in filter_sizes:
            input_length = (input_length - filter_size + stride) // stride
        return input_length

    return get_output_length(width), get_output_length(height) 

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # identity block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(FixedBatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def nn_base(img_input, trainable=False):

    # # Determine proper input shape
    # if K.image_dim_ordering() == 'th':
    #     input_shape = (3, None, None)
    # else:
    #     input_shape = (None, None, 3)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    #
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    net1 = ZeroPadding2D((3, 3))(img_input)

    net1 = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(net1)
    net1 = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(net1)
    net1 = Activation('relu')(net1)
    net1 = MaxPooling2D((3, 3), strides=(2, 2))(net1)

    net2 = conv_block(net1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    net2 = identity_block(net2, 3, [64, 64, 256], stage=2, block='b', trainable=trainable)
    # net2 = identity_block(net2, 3, [64, 64, 256], stage=2, block='c', trainable=trainable)

    net3 = conv_block(net2, 3, [128, 128, 512], stage=3, block='a', trainable=trainable)
    net3 = identity_block(net3, 3, [128, 128, 512], stage=3, block='b', trainable=trainable)
    # net3 = identity_block(net3, 3, [128, 128, 512], stage=3, block='c', trainable=trainable)
    # net3 = identity_block(net3, 3, [128, 128, 512], stage=3, block='d', trainable=trainable)

    net4 = conv_block(net3, 3, [256, 256, 1024], stage=4, block='a', trainable=trainable)
    net4 = identity_block(net4, 3, [256, 256, 1024], stage=4, block='b', trainable=trainable)
    # net4 = identity_block(net4, 3, [256, 256, 1024], stage=4, block='c', trainable=trainable)
    # net4 = identity_block(net4, 3, [256, 256, 1024], stage=4, block='d', trainable=trainable)
    # net4 = identity_block(net4, 3, [256, 256, 1024], stage=4, block='e', trainable=trainable)
    # net4 = identity_block(net4, 3, [256, 256, 1024], stage=4, block='f', trainable=trainable)
    return [net3, net4]


# def classifier_layers(x, input_shape, trainable=False):
#
#     # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
#     # (hence a smaller stride in the region that follows the ROI pool)
#     if K.backend() == 'tensorflow':
#         x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
#     elif K.backend() == 'theano':
#         x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(1, 1), trainable=trainable)
#
#     x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
#     x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)
#     x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)
#
#     return x
class SSDModel(object):
    def __init__(self, scales, aspect_ratios, n_classes):
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.n_classes = n_classes+1
    def __call__(self, x):
        base_layers = nn_base(x, trainable=True)
        n_boxes = len(self.scales) * len(self.aspect_ratios)
        net3 = base_layers[0];net4 = base_layers[1]
        net3 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(net3)
        net4 = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(net4)
        classes3 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), kernel_initializer='uniform', padding="same",name='classes3', activation='sigmoid')(net3)
        classes4 = Conv2D(n_boxes * self.n_classes, (1, 1), strides=(1, 1), kernel_initializer='uniform',padding="same", name='classes4', activation='sigmoid')(net4)
        classes3_reshaped = Reshape((-1, self.n_classes), name='classes3_reshaped')(classes3)
        classes4_reshaped = Reshape((-1, self.n_classes), name='classes4_reshaped')(classes4)
        offset3 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same", kernel_initializer='zero', name='offset3',activation='linear')(net3)
        offset4 = Conv2D(n_boxes * 4, (1, 1), strides=(1, 1), padding="same", kernel_initializer='zero', name='offset4',activation='linear')(net4)
        offset3_reshaped = Reshape((-1, 4), name='offset3_reshaped')(offset3)
        offset4_reshaped = Reshape((-1, 4), name='offset4_reshaped')(offset4)

        anchors3 = AnchorBoxes(img_height=128, img_width=59, aspect_ratios=self.aspect_ratios, scales=self.scales)(offset3)
        anchors4 = AnchorBoxes(img_height=128, img_width=59, aspect_ratios=self.aspect_ratios, scales=self.scales)(offset4)
        anchors3_reshaped = Reshape((-1, 4), name='anchors3_reshaped')(anchors3)
        anchors4_reshaped = Reshape((-1, 4), name='anchors4_reshaped')(anchors4)
        classes = Concatenate(axis=1, name='classes_concat')([classes3_reshaped, classes4_reshaped])
        anchors = Concatenate(axis=1, name='anchors_concat')([anchors3_reshaped, anchors4_reshaped])
        offset = Concatenate(axis=1, name='offset_concat')([offset3_reshaped, offset4_reshaped])
        # output = Concatenate(axis=2, name='output')([classes, offset])
        # layers = [offset3, offset4]
        return classes, offset, anchors



def rpn(base_layers, num_anchors):

    x = Convolution2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Convolution2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Convolution2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]

# def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):
#
#     # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround
#
#     if K.backend() == 'tensorflow':
#         pooling_regions = 14
#         input_shape = (num_rois, 14, 14, 1024)
#     elif K.backend() == 'theano':
#         pooling_regions = 7
#         input_shape = (num_rois, 1024, 7, 7)
#
#     out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
#     out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
#
#     out = TimeDistributed(Flatten())(out)
#
#     out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
#     # note: no regression target for bg class
#     out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)
#     return [out_class, out_regr]

