from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense
from RoiPooling import RoiPooling
import keras as k
import tensorflow as tf



class FootNet_v3(object):
    def __init__(self, aspect_ratio, scales):
        self.aspect_ratio = aspect_ratio
        self.scales = scales

    def base_net(self, x):
        init = k.initializers.glorot_normal()
        net1 = Conv2D(64, (7, 7), padding='same', strides=[2, 2],
            kernel_initializer=init, activation='relu')(x)
        net2 = Conv2D(128, (3, 3), padding='same', strides=[1, 1], activation='relu',
            kernel_initializer=init)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net2)
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
            kernel_initializer=init)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net3)
        net4 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=init)(net3)
        net4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(net4)
        net4 = Conv2D(256, (3, 3), padding='same', strides=[1, 1], activation='relu',
                      kernel_initializer=init)(net4)
        return net4


    def RPN(self, base_net, trainable=True):
        init = k.initializers.glorot_normal()
        num_boxes = len(self.scales) * len(self.aspect_ratio)
        net1 = Conv2D(512, (3, 3), padding='same', strides=[1, 1], kernel_initializer=init,
                      activation='relu', trainable=trainable)(base_net)
        classes = Conv2D(num_boxes * 2, (1, 1), padding='same', strides=[1, 1], kernel_initializer='glorot_uniform',
                      activation='sigmoid', trainable=trainable)(net1)
        classes = Conv2D(num_boxes * 2, (1, 1), padding='same', strides=[1, 1], kernel_initializer='glorot_uniform',
                      activation='sigmoid', trainable=trainable)(classes)
        offset = Conv2D(num_boxes * 4, (1, 1), padding='same', strides=[1, 1], kernel_initializer='zeros',
                      activation='tanh', trainable=trainable)(base_net)
        offset = Conv2D(num_boxes * 4, (1, 1), padding='same', strides=[1, 1], kernel_initializer='zeros',
                         activation='tanh', trainable=trainable)(offset)
        classes_reshape = tf.reshape(classes, [-1, 2])
        classes_reshape = tf.nn.softmax(classes_reshape)
        offset_reshape = tf.reshape(offset, [-1, 4])
        return classes_reshape, offset_reshape


    def classcify(self, base_net, rois):
        pooling_layer = RoiPooling(pool_size=14)(base_net, rois)  #[batch_size, pool_size, pool_size, channels]
        net = Conv2D(256, (3, 3), padding='same', strides=[1, 1], kernel_initializer='zeros',
                                  activation='tanh')(pooling_layer)




        return





    #
    # def classcify(self, base_net, rois, out_size, trainable):
    #     init = k.initializers.glorot_normal()
    #     net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
    #     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
    #               kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    #     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
    #                kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    #     net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
    #     net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
    #     cls = Dense(2, activation='sigmoid', trainable=trainable)(net1)
    #     return cls
    #
    #
    # def box_regressor(self, base_net, rois, out_size, trainable):
    #     init = k.initializers.glorot_uniform()
    #     net1 = RoiLayer(out_size=out_size, rois=rois)(base_net)
    #     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
    #               kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    #     net1 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],
    #               kernel_initializer=init, activation='relu', trainable=trainable)(net1)
    #     net1 = tf.reshape(net1, [-1, out_size[0] * out_size[1] * 256])
    #     net1 = Dense(4096, activation='relu', trainable=trainable)(net1)
    #     box = Dense(4, activation='sigmoid', trainable=trainable)(net1)
    #     return box
    #
