import os
import cv2
import scipy.io as sio
import numpy as np
###############################################
##########  File I/O funcnctions   ############
###############################################
def source_img_resize(input_dir, output_dir, shape):
    # This function used for resize img and save to another folder
    # input_dir: source img folder
    # shape: A tuple shaped (columns, rows)
    file_path = []; output_path = []
    IOfilelist(input_dir, file_path, output_dir, output_path)
    mkdir(input_dir, output_dir)
    for index, path_i in enumerate(file_path):
        if index % 10 == 0:
          print('complete ', index/len(file_path))
        img = cv2.imread(path_i)
        img = cv2.resize(img, shape)
        cv2.imwrite(output_path[index], img)
    return
def IOfilelist(input_dir, file_path, output_dir, output_path):
    # This function used for create input file path list and output file path list
    # input_dir: The source file folder path
    # file_path: A void list, used for storage input file path list
    # output_path: A void list, used for storage output file path list
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        outfile = os.path.join(output_dir, sub_path)
        if os.path.isdir(filename):
            IOfilelist(filename, file_path, outfile, output_path)
        else:
            file_path.append(filename)
            output_path.append(outfile)
def getfilelist(input_dir, file_path):
    # this function used for return all file's path in current folder and sub folder
    # input_dir: A string represent folder path
    # file_path: A void list used for store file's path
    for sub_path in os.listdir(input_dir):
        filename = os.path.join(input_dir, sub_path)
        if os.path.isdir(filename):
            getfilelist(filename, file_path)
        else:
            file_path.append(filename)
def mkdir(path, output):
    # This function used for create sub folder from source folder to target.
    # path: This is the source folder
    # output: This is target folder.
    # Warning: This output folder must be established before using this function
    for sub_path in os.listdir(path):
        filepath = os.path.join(path, sub_path)
        outpath = os.path.join(output, sub_path)
        if os.path.isdir(filepath):
            if not os.path.exists(outpath):
                os.mkdir(outpath)
            mkdir(filepath, outpath)
#####################################################################################
#####################################################################################
def create_train_test_table(input_dir,label_dir,train_key, output_train, output_test):
    # This function used for create two txt files, which contains img file path and label file path
    # The two txt files are train table and test table
    # Each table have two columns,the first column is img path, the second column is label path
    # input_dir, label_dir: The img source folder path, the label source folder path
    # train_key: A int number, which is represent key point between train and test
    # output_train, output_test: The txt file output path
    p_id_img = os.listdir(input_dir)
    p_id_lab = os.listdir(label_dir)
    p_id_img.sort(key=lambda x: int(x))
    p_id_lab.sort(key=lambda x: int(x))
    for ind_pid in range(len(p_id_img)):
      if ind_pid <= train_key:
        path_img = os.path.join(input_dir, p_id_img[ind_pid])
        path_lab = os.path.join(label_dir, p_id_lab[ind_pid])
        list_img = []; list_lab = []
        getfilelist(path_img, list_img)
        getfilelist(path_lab, list_lab)
        train_tab = open(output_train, 'a')
        for ind_fid in range(len(list_img)):
            train_tab.write(list_img[ind_fid])
            train_tab.write(';')
            train_tab.write(list_lab[ind_fid])
            train_tab.write('\n')
        train_tab.close()
      else:
          path_img = os.path.join(input_dir, p_id_img[ind_pid])
          path_lab = os.path.join(label_dir, p_id_lab[ind_pid])
          list_img = []
          list_lab = []
          getfilelist(path_img, list_img)
          getfilelist(path_lab, list_lab)
          test_tab = open(output_test, 'a')
          for ind_fid in range(len(list_img)):
              test_tab.write(list_img[ind_fid])
              test_tab.write(';')
              test_tab.write(list_lab[ind_fid])
              test_tab.write('\n')
          test_tab.close()
    return
#####################################################################################
#####################################################################################
#####################################################################################
def load_data(train_path, test_path):
    # This function used for load data
    f_train = open(train_path)
    f_test = open(test_path)
    train_line = f_train.readlines()
    test_line = f_test.readlines()
    train_x = []; train_roi = []; test_x = []; test_roi = []
    for line_i in train_line:
       s_line = line_i.split(";", 2)
       img = cv2.imread(s_line[0])
       lab = sio.loadmat(s_line[1][0:len(s_line[1])-1])['label'][0, :]
       train_x.append(img)
       train_roi.append(lab)
    train_x = np.array(train_x)
    train_roi = np.array(train_roi)
    for line_i in test_line:
       s_line = line_i.split(";", 2)
       img = cv2.imread(s_line[0])
       lab = sio.loadmat(s_line[1][0:len(s_line[1])-1])['label'][0, :]
       test_x.append(img)
       test_roi.append(lab)
    test_x = np.array(test_x)
    test_roi = np.array(test_roi)
    return train_x, train_roi, test_x, test_roi



if __name__ == '__main__':
    # unit testing interference
    # train_txt = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\V1.0.0.0_128\\train.txt'
    # test_txt = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\V1.0.0.0_128\\test.txt'
    # train_x, train_roi, test_x, test_roi = load_data(train_txt, test_txt)
    # print(np.shape(train_x))
    # print(np.shape(train_x))
    # print(np.shape(train_roi))
    # print(np.shape(test_x))
    # print(np.shape(test_roi))



    input_dir = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\mini_V1.0.0.0_128\image'
    lab_dir = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\mini_V1.0.0.0_128\label'
    output_train = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_train.txt'
    output_test = 'E:\PROJECT\\barefoot_fast_rcnn\data_txt\\mini_test.txt'
    create_train_test_table(input_dir, lab_dir, 1, output_train, output_test)

    #print(file_list)

    # output_dir = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\\V1.0.0.0_128'
    # source_img_resize(input_dir, output_dir, (59, 128))

    #
    # mkdir(path=input_dir, output=output_dir)
    # file_path = []; output_path = []
    # IOfilelist(input_dir, file_path, output_dir, output_path)
    # print(file_path)
    # print(len(file_path))
    # print(len(output_path))
