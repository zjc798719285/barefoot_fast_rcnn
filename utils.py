import os
import cv2
import scipy.io as sio

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
####################################################################################
def get_filepath_table():
    return








if __name__ == '__main__':
    input_dir = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\\V1.0.0.0'
    output_dir = 'E:\PROJECT\Foot_Height\data_Foot_Height\\barefoot_standard\RCNN\\V1.0.0.0_128'
    source_img_resize(input_dir, output_dir, (59, 128))

    #
    # mkdir(path=input_dir, output=output_dir)
    # file_path = []; output_path = []
    # IOfilelist(input_dir, file_path, output_dir, output_path)
    # print(file_path)
    # print(len(file_path))
    # print(len(output_path))
