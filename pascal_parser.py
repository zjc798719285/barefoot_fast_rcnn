import os
import xml.etree.ElementTree as ET
import numpy as np
from AnchorBoxes import convert_coordinates

def get_data(input_dir):
    train_list = [];test_list = []
    train_files = []; test_files = []
    train_txt = os.path.join(input_dir, 'ImageSets', 'Main', 'train.txt')
    test_txt = os.path.join(input_dir, 'ImageSets', 'Main', 'val.txt')
    f = open(train_txt, 'r')
    for line in f:
        train_files.append(line.strip() + '.jpg')
    f = open(test_txt, 'r')
    for line in f:
        test_files.append(line.strip() + '.jpg')
    for train_i in train_files:
        annotation = os.path.join(input_dir, 'Annotations', train_i[0:-4] + '.xml')
        image_path = os.path.join(input_dir, 'JPEGImages', train_i)
        et = ET.parse(annotation)
        root = et.getroot()
        objs = root.findall('object')
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        obj_list = []
        for obj_i in objs:
            name = obj_i.find('name').text
            xmin = int(round(float(obj_i.find('bndbox').find('xmin').text)))
            xmax = int(round(float(obj_i.find('bndbox').find('xmax').text)))
            ymin = int(round(float(obj_i.find('bndbox').find('ymin').text)))
            ymax = int(round(float(obj_i.find('bndbox').find('ymax').text)))
            rect_center = np.array([xmin, ymin, xmax, ymax])
            rect_corner = convert_coordinates(anchor=rect_center, img_width=img_width, img_height=img_height)
            obj_dict = {'obj_name': name, 'rect_center': rect_center, 'rect_corner': rect_corner}
            obj_list.append(obj_dict)
        file_info = {'Image': image_path, 'width': img_width, 'height': img_height,
                     'obj': obj_list}
        train_list.append(file_info)
 #############################################################################
    for test_i in test_files:
        annotation = os.path.join(input_dir, 'Annotations', test_i[0:-4] + '.xml')
        image_path = os.path.join(input_dir, 'JPEGImages', test_i)
        et = ET.parse(annotation)
        root = et.getroot()
        objs = root.findall('object')
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)
        obj_list = []
        for obj_i in objs:
            name = obj_i.find('name').text
            xmin = int(round(float(obj_i.find('bndbox').find('xmin').text)))
            xmax = int(round(float(obj_i.find('bndbox').find('xmax').text)))
            ymin = int(round(float(obj_i.find('bndbox').find('ymin').text)))
            ymax = int(round(float(obj_i.find('bndbox').find('ymax').text)))
            rect_center = np.array([xmin, ymin, xmax, ymax])
            rect_corner = convert_coordinates(anchor=rect_center, img_width=img_width, img_height=img_height)
            obj_dict = {'obj_name': name, 'rect_center': rect_center, 'rect_corner': rect_corner}
            obj_list.append(obj_dict)
        file_info = {'Image': image_path, 'width': img_width, 'height': img_height,
                     'obj': obj_list}
        test_list.append(file_info)
    return train_list, test_list

if __name__ == '__main__':
    input_path = 'E:\PROJECT\keras-frcnn\VOCtrainval_11-May-2012\VOC2012'
    train_list, test_list = get_data(input_path)
    print()