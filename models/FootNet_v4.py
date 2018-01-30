from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense,Flatten
from roi_pooling import RoiLayer
import keras as k
import tensorflow as tf
class FootNet_v4(object):
    def __init__(self):
        return

    def base_net(self, x, trainable):
        init = k.initializers.glorot_normal()
        net1 = Conv2D(64, (3, 3), padding='same', strides=[2, 2],
            kernel_initializer=init, activation='relu', trainable=trainable)(x)
        net1 = Conv2D(64, (3, 3), padding='same', strides=[1, 1],
                      kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net2 = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(net1)
        net3 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
            kernel_initializer=init, trainable=trainable)(net2)
        return net3

    def classcify(self, base_net, rois, out_size, trainable):
        init = k.initializers.glorot_normal()
        net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
        net1 = Conv2D(128, (3, 3), padding='same', strides=(1, 1),
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = MaxPooling2D((3, 3), padding='same', strides=(2, 2))(net1)
        net1 = Conv2D(256, (3, 3), padding='same', strides=(1, 1),
                   kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = Flatten()(net1)
        net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
        cls = Dense(2, activation='sigmoid', trainable=trainable)(net1)
        return cls


    def box_regressor(self, base_net, rois, out_size, trainable):
        init = k.initializers.glorot_uniform()
        net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
        net1 = Conv2D(128, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = MaxPooling2D((3, 3), padding='same', strides=[2, 2])(net1)
        net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
        net1 = Flatten()(net1)
        net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
        box = Dense(4, activation='sigmoid', trainable=trainable)(net1)
        return box

