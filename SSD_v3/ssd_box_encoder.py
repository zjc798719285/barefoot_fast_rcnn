import numpy as np

def convert_coordinates(tensor,img_height, img_width):
    #Return a ndarray
    tensor1 = np.copy(tensor).astype(np.float)
    tensor2 = np.copy(tensor).astype(np.float)
    #中心坐标转化为[xmin,ymin,xmax,ymax]坐标，并进行限制
    tensor1[..., 0] = tensor[..., 0] - tensor[..., 3] / 2  #set xmin
    tensor1[..., 1] = tensor[..., 1] - tensor[..., 2] / 2  #set ymin
    tensor1[..., 2] = tensor[..., 0] + tensor[..., 3] / 2  #set xmax
    tensor1[..., 3] = tensor[..., 1] + tensor[..., 2] / 2  #set ymax
    tensor1[tensor1[..., 0] < 0] = 0; tensor1[tensor1[..., 0] > img_height] = img_height  # 0<xmin<img_heighit
    tensor1[tensor1[..., 1] < 0] = 0; tensor1[tensor1[..., 1] > img_width] = img_width    # 0<ymin<img_width
    tensor1[tensor1[..., 2] < 0] = 0; tensor1[tensor1[..., 2] > img_height] = img_height  # 0<xmax<img_heighit
    tensor1[tensor1[..., 3] < 0] = 0; tensor1[tensor1[..., 3] > img_width] = img_width    # 0<ymax<img_width
    tensor2[..., 0] = tensor1[..., 0] / img_height
    tensor2[..., 1] = tensor1[..., 1] / img_width
    tensor2[..., 2] = (tensor1[..., 3] - tensor1[..., 1]) / img_width
    tensor2[..., 3] = (tensor1[..., 2] - tensor1[..., 0]) / img_height
    return tensor2


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


def ssd_box_encoder_one_image(roi_list, classes_list, num_classes,
                              anchors, iou_thresh_pos, iou_thresh_neg):
    eps = 1e-6
    (num_boxes, channels) = np.shape(anchors)
    y_classses = np.zeros((1, num_boxes, num_classes + 1))  # '+1'代表背景类别
    y_anchors = np.zeros((1, num_boxes, 4))
    for id_anc, anchor in enumerate(anchors):
        for roi_i, class_i in zip(roi_list, classes_list):
            if iou_eval(roi_i, anchor) >= iou_thresh_pos:
                y_anchors[0, id_anc, 0] = roi_i[0] - anchor[0]
                y_anchors[0, id_anc, 1] = roi_i[1] - anchor[1]
                y_anchors[0, id_anc, 2] = roi_i[2] - anchor[2]
                y_anchors[0, id_anc, 3] = roi_i[3] - anchor[3]
            if iou_eval(roi_i, anchor) >= iou_thresh_pos:
                y_classses[0, id_anc, int(class_i)] = 1
            elif iou_eval(roi_i, anchor) < iou_thresh_neg:
                y_classses[0, id_anc, 0] = 1
    return y_anchors, y_classses


def ssd_box_encoder_batch(roi_list, classes_list, num_classes,
                             anchors, iou_thresh_pos, iou_thresh_neg):
      (batch, num_boxes, channels) = np.shape(anchors)
      y_classses = np.zeros((batch, num_boxes, num_classes + 1))  # '+1'代表背景类别
      y_anchors = np.zeros((batch, num_boxes, 4))
      for i in range(batch):
          y_anchors[i, :, :], y_classses[i, :, :] = ssd_box_encoder_one_image(
                                                             roi_list=roi_list[i],
                                                             classes_list=classes_list[i],
                                                             num_classes=num_classes,
                                                             anchors=anchors[i, :, :],
                                                             iou_thresh_pos=iou_thresh_pos,
                                                             iou_thresh_neg=iou_thresh_neg)
      return y_classses, y_anchors














if __name__ == '__main__':
    roi_list = [np.array([[0.1, 0.2, 0.1, 0.1]])]
    classes_list = [np.array([[1]])]
    anchors = np.zeros([1, 20646, 4])
    y_classes, y_anchors = ssd_box_encoder_batch(roi_list=roi_list,
                                                 classes_list=classes_list,
                                                 anchors=anchors,
                                                 iou_threshold=-0.5,
                                                 num_classes=1)
    print(np.shape(y_classes))
    print(np.shape(y_anchors))






