import numpy as np

def roi_prop_generator(scale_r, scale_c, num_step):
    # for testing
    rois=[]
    for scale_r_i in scale_r:
        for scale_c_i in scale_c:
             base_roi = np.reshape(a=np.array([0, 0, scale_r_i, scale_c_i]), newshape=(1, 4))
             roi_i = img_get_roi(base_roi, num_step)
             rois.append(roi_i)
    rois = np.array(np.reshape(a=rois, newshape=(-1, 4)))
    return rois

def img_get_roi(base_roi, num_step):
    rois = []
    step_r = (1-base_roi[0, 3])/num_step
    step_c = (1-base_roi[0, 2])/num_step
    for i in range(num_step):
        for j in range(num_step):
            rois.append(np.array([base_roi[0, 0]+i*step_r, base_roi[0, 1]+j*step_c,
                            base_roi[0, 2], base_roi[0, 3]]))
    return np.array(rois)


def pos_neg_roi_generator(rois, cls, num_rois, num_cls):
    rois = np.array(np.reshape(a=rois, newshape=(-1, 4)))
    cls = np.array(np.reshape(a=cls, newshape=(-1, 1)))
    cls_label = []; cls_rois = []; regre_rois= []; regre_gt= []
    lab = np.zeros([1, num_cls + 1])
    for roi_i, cls_i in zip(rois, cls):
        for i in range(num_rois):
            # positive
            h_expand = np.random.uniform(low=1, high=1.1)  # set high parameter
            w_expand = np.random.uniform(low=1, high=1.1)
            x_up_off = np.random.uniform(low=-(h_expand - 1) * roi_i[3], high=0)
            y_up_off = np.random.uniform(low=-(w_expand - 1) * roi_i[2], high=0)
            roi_i_pos = np.array([max(roi_i[0] + x_up_off, 0), max(roi_i[1] + y_up_off, 0),
                          min(roi_i[2] * w_expand, 1), min(roi_i[3] * h_expand, 1)])
            label_pos = lab.copy(); label_pos[0, int(cls_i)] = 1
            cls_label.append(label_pos); cls_rois.append(roi_i_pos)
            regre_rois.append(roi_i_pos); regre_gt.append(roi_i)
            # negtive
            border_top = roi_i[0]; border_bottom = 1 - roi_i[0] - roi_i[2]
            border_left = roi_i[1]; border_right = 1 - roi_i[1] - roi_i[3]
            x_offset = np.random.uniform(low=-border_top, high=border_bottom)
            y_offset = np.random.uniform(low=-border_left, high=border_right)
            h_zoom = np.random.uniform(low=0.5, high=1.3)
            w_zoom = np.random.uniform(low=0.5, high=1.3)
            roi_neg = np.array([roi_i[0] + x_offset, roi_i[1]
                                + y_offset, roi_i[2] * w_zoom, roi_i[3] * h_zoom])
            label_neg = lab.copy(); label_neg[0, num_cls] = 1
            cls_label.append(label_neg); cls_rois.append(roi_neg)
    cls_label = np.array(np.reshape(a=cls_label, newshape=(-1, num_cls+1)))
    cls_rois = np.array(np.reshape(a=cls_rois, newshape=(-1, 4)))
    return cls_rois, cls_label, regre_rois, regre_gt


def cls_roi_generator(roi, num_rois, num_cls):
   roi = np.reshape(a=roi, newshape=[num_cls, 4])
   cls_label = []; cls_roi = []; cls_gt_roi = []
   for ind in range(num_cls):
       for i in range(num_rois):
           h_expand = np.random.uniform(low=1, high=1.2)  # set high parameter                  :
           w_expand = np.random.uniform(low=1, high=1.2)
           x_offset = np.random.uniform(low=-(h_expand - 1) * roi[ind, 2], high=0)
           y_offset = np.random.uniform(low=-(w_expand - 1) * roi[ind, 3], high=0)
           roi_i_prop = [max(roi[ind, 0] + x_offset, 0), max(roi[ind, 1] + y_offset, 0),
                         min(roi[ind, 2] * h_expand, 1), min(roi[ind, 3] * w_expand, 1)]
           cls_roi.append(roi_i_prop); cls_gt_roi.append(roi[ind, :])
           lab = np.zeros(shape=[num_cls + 1]); lab[ind] = 1; cls_label.append(lab)
   for i in range(num_rois):
       x = np.random.uniform(low=0, high=1);y = np.random.uniform(low=0, high=1)
       h = 1 - x; w = 1 - y
       cls_roi.append([x, y, h, w]); cls_gt_roi.append(np.zeros(shape=[4]))
       lab = np.zeros(shape=[num_cls + 1]);lab[num_cls] = 1; cls_label.append(lab)
   return cls_roi, cls_label, cls_gt_roi

    # for ind, cls_i in enumerate(cls):
    #     if cls_i[0] > cls_i[1]:
    #         cls_roi.append(rois[ind])
    #         cls_prob.append(cls_i[0])
    # assert len(cls_prob) == len(cls_roi)
    # cls_prob_reshape = np.reshape(a=cls_prob, newshape=(1, len(cls_prob)))
    # norm_cls_prob = cls_prob_reshape / sum(cls_prob_reshape)
    # for ind, cls_roi_i in enumerate(cls_roi):
    #     final_roi = final_roi + cls_roi_i * norm_cls_prob[0, ind]
    # return final_roi
    #

def iou_eval(gt, dr):
    # gt: GroundTruth roi
    # dr: DetectionResult roi
    # return overlap ratio of gt and dr
 gt = np.reshape(a=gt, newshape=(4, 1))
 dr = np.reshape(a=dr, newshape=(4, 1))
 start_x_gt = gt[0]; end_x_gt = gt[0] + gt[2]
 start_y_gt = gt[1]; end_y_gt = gt[1] + gt[3]
 start_x_dr = dr[0]; end_x_dr = dr[0] + dr[2]
 start_y_dr = dr[1]; end_y_dr = dr[1] + dr[3]
 if (end_x_gt <= start_x_dr or end_x_dr <= start_x_gt or
     end_y_gt <= start_y_dr or end_y_dr <= start_y_gt):
     return 0
 else:
     gt_area = gt[2] * gt[3]; dr_area = dr[2] * dr[3]
     start_x = max(start_x_gt, start_x_dr); end_x = min(end_x_gt, end_x_dr)
     start_y = max(start_y_gt, start_y_dr); end_y = min(end_y_gt, end_y_dr)
     h_inter = end_x - start_x; w_inter = end_y - start_y
     inter_area = h_inter * w_inter
     union_area = gt_area + dr_area - inter_area
     return inter_area / union_area

if __name__ == '__main__':
    rois = np.array([0.1, 0.1, 0.4, 0.3])
    cls_roi, cls_label, regre_rois, regre_gt = pos_neg_roi_generator(rois, [0], 10, 1)
    print(np.shape(cls_roi), np.shape(regre_rois))
    print(np.shape(cls_label), np.shape(regre_gt))
    print(cls_label)
    print(cls_roi)
    # scale_r = np.array([0.7, 0.8, 0.9, 0.95, 0.98])
    # scale_c = np.array([0.7, 0.8, 0.9, 0.95, 0.98])
    # num_step = 5
    # rois = roi_prop_generator(scale_r, scale_c, num_step)
    # print(np.shape(rois))
    # rois = np.array([[0.1, 0.1, 0.3, 0.4], [0.5, 0.1, 0.9, 0.4], [0.1, 0.1, -0.3, 0.4]])
    # print(rois)
    # roi = roi_check(rois)
    # print(roi)
    #
    # image1 = cv2.imread('E:\PROJECT\Data_annotation\\New Folder\\0101.jpg')
    # image2 = cv2.imread('E:\PROJECT\Data_annotation\\New Folder\\4284.jpg')
    # print(np.shape(image1))
    # # cv2.imshow('image1', image1)
    # # cv2.waitKey(0)
    # roi1 = np.random.uniform(low=0, high=1, size=[1, 4])
    # # print(np.shape(roi1))
    # cls_roi, cls_label, cls_gt_roi = cls_roi_generator(roi=roi1, num_rois=10, num_cls=1)
    # # print(cls_roi)
    # # print(cls_label)
    # # print(cls_gt_roi)
    # # print(np.shape(cls_roi))
    # # print(np.shape(cls_label))
    # # print(np.shape(cls_gt_roi))






















   # print(np.shape(label))

