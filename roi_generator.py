import numpy as np


def cls_roi_generator(roi, num_rois, num_cls):
    roi = np.reshape(a=roi, newshape=[num_cls, 4])
    cls_label = []; cls_roi = []; cls_gt_roi = []
    for ind in range(num_cls):
        for i in range(num_rois):
            h_expand = np.random.uniform(low=1, high=1.2) # set high parameter
            w_expand = np.random.uniform(low=1, high=1.2)
            x_offset = np.random.uniform(low=-(h_expand - 1) * roi[ind, 2], high=0)
            y_offset = np.random.uniform(low=-(w_expand - 1) * roi[ind, 3], high=0)
            roi_i_prop = [max(roi[ind, 0] + x_offset, 0), max(roi[ind, 1] + y_offset, 0),
                          min(roi[ind, 2] * h_expand, 1), min(roi[ind, 3] * w_expand, 1)]   #确保矩形框不超过0~1
            cls_roi.append(roi_i_prop); cls_gt_roi.append(roi[ind, :])
            lab = np.zeros(shape=[num_cls + 1]); lab[ind] = 1; cls_label.append(lab)
    for i in range(num_rois):
        x = np.random.uniform(low=0, high=1);y = np.random.uniform(low=0, high=1)
        h = 1 - x; w = 1 - y
        cls_roi.append([x, y, h, w]); cls_gt_roi.append(np.zeros(shape=[4]))
        lab = np.zeros(shape=[num_cls + 1]);lab[num_cls] = 1; cls_label.append(lab)
    return cls_roi, cls_label, cls_gt_roi


def roi_filter(rois, cls):
    if len(rois) < 1:
        return [0, 0, 1, 1]
    assert len(rois) == len(cls)
    final_roi = np.zeros([1, 4])
    cls_roi = []; cls_prob = []
    for ind, cls_i in enumerate(cls):
        if cls_i[0] > cls_i[1]:
            cls_roi.append(rois[ind])
            cls_prob.append(cls_i[0])
    assert len(cls_prob) == len(cls_roi)
    norm_cls_prob = np.reshape(a=cls_prob / sum(cls_prob), newshape=(1, len(cls_prob)))
    for ind, cls_roi_i in enumerate(cls_roi):
        final_roi = final_roi + cls_roi_i * norm_cls_prob[0, ind]
    return final_roi


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
    # image1 = cv2.imread('E:\PROJECT\Data_annotation\\New Folder\\0101.jpg')
    # image2 = cv2.imread('E:\PROJECT\Data_annotation\\New Folder\\4284.jpg')
    # print(np.shape(image1))
    # cv2.imshow('image1', image1)
    # cv2.waitKey(0)
    roi1 = np.random.uniform(low=0, high=1, size=[1, 4])
    # print(np.shape(roi1))
    cls_roi, cls_label, cls_gt_roi = cls_roi_generator(roi=roi1, num_rois=10, num_cls=1)
    # print(cls_roi)
    # print(cls_label)
    # print(cls_gt_roi)
    # print(np.shape(cls_roi))
    # print(np.shape(cls_label))
    # print(np.shape(cls_gt_roi))






   #
   # roi2 = [0.2, 0.2, 0.2, 0.2]
   # roi3 = [0.3, 0.2, 0.2, 0.2]
   # roi4 = [0.2, 0.3, 0.2, 0.2]
   # print(iou_eval(roi1, roi2))
   # print(iou_eval(roi1, roi3))
   # print(iou_eval(roi1, roi4))
   #
   # roi = [0.2, 0.1, 0.5, 0.5]
   # rois, label, RPN_rois = pos_neg_roi_generator(roi, 10)
   # print(np.shape(RPN_rois))
   # # box_roi = box_roi_generator(cls_label=label, roi=roi)
   # #
   # for box_roi_i in box_roi:
   #     print(box_roi_i)
   # print(np.shape(rois))
   # print(np.shape(label))

