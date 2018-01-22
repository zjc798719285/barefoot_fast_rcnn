from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, SeparableConv2D
from keras.layers import Dense, BatchNormalization, Dropout, Activation, regularizers
from roi_pooling import roi_layer, RoiLayer
import keras as k
import tensorflow as tf

def base_net(x, trainable):
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

def classcify(base_net, rois, out_size, trainable):
     init = k.initializers.glorot_normal()
     net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                   kernel_initializer=init, activation='relu', trainable=trainable)(net1)
     net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
     net1 = Dense(4096, activation='tanh', trainable=trainable)(net1)
     cls = Dense(2, trainable=trainable)(net1)
     return cls

def box_regressor(base_net, rois,out_size, trainable):
    init = k.initializers.glorot_uniform()
    net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
    net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
                  kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
    net1 = Dense(4096, activation='tanh', trainable=trainable)(net1)
    box = Dense(4, trainable=trainable)(net1)
    return box

