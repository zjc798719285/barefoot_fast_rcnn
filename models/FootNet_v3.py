from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense, BatchNormalization, Dropout, Activation, regularizers
from roi_pooling import RoiLayer
import keras as k
import tensorflow as tf
class FootNet_v3(object):
    def __init__(self):
        return

    def base_net(self, x, trainable):
        init = k.initializers.glorot_normal()
        net1 = Conv2D(64, (3, 3), padding='same', strides=[2, 2],
            kernel_initializer=init, activation='relu', trainable=trainable)(x)
        net2 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
            kernel_initializer=init, trainable=trainable)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net2)
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
            kernel_initializer=init, trainable=trainable)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net3)
        return net3


    def RPN(self, base_net, out_size, trainable, num_rois):
        init = k.initializers.glorot_normal()
        net1 = tf.image.resize_images(base_net, tuple(out_size))
        net1 = Conv2D(num_rois, (3, 3), padding='same', strides=[1, 1],
                      kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1]])
        net1 = Dense(32, activation='relu', trainable=trainable)(net1)
        RPN_rois = Dense(4, activation='sigmoid', trainable=trainable)(net1)
        return RPN_rois


    def classcify(self, base_net, rois, out_size, trainable):
        init = k.initializers.glorot_normal()
        net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
        net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                   kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
        net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
        cls = Dense(2, activation='sigmoid', trainable=trainable)(net1)
        return cls


    def box_regressor(self, base_net, rois, out_size, trainable):
        init = k.initializers.glorot_uniform()
        net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
        net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
        net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
        box = Dense(4, activation='sigmoid', trainable=trainable)(net1)
        return box

