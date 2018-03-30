import resnet
import tensorflow as tf
import numpy as np
from box_filter import box_filter, rect_iou, class_acc
from box_encoder_decoder import ssd_box_encoder_batch
from BatchGenerator import BatchGenerator, load_data
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adadelta
import KerasLoss

######################
# Parameters setting #
######################
train_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
test_txt = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\train.txt'
batch_size = 10
num_boxes_one_image = 1248
pos_neg_ratio = 10
#############
# Load Data #
#############
Image = Input(shape=[128, 59, 3])

model_SSD = Model(Image, SSD)
optimizer = Adadelta(lr=0.01, rho=0.9)
model_SSD.compile(optimizer=optimizer, loss=KerasLoss.loss)
train_x, train_roi, test_x, test_roi, train_cls, test_cls = load_data(train_txt, test_txt)
trainData = BatchGenerator(image=train_x, roi=train_roi, classes=train_cls, batch_size=batch_size)
train_x, train_roi_list, train_class_list = trainData.next_batch()
s1 = model_SSD.predict_on_batch(train_x)
print()







if __name__ == '__main__':
     for i in range(10):
         print(i)