import numpy as np
#
# def convert_coordinates(tensor,img_height, img_width):
#     #Return a ndarray
#     tensor1 = np.copy(tensor).astype(np.float)
#     tensor2 = np.copy(tensor).astype(np.float)
#     #中心坐标转化为[xmin,ymin,xmax,ymax]坐标，并进行限制
#     tensor1[..., 0] = tensor[..., 0] - tensor[..., 3] / 2  #set xmin
#     tensor1[..., 1] = tensor[..., 1] - tensor[..., 2] / 2  #set ymin
#     tensor1[..., 2] = tensor[..., 0] + tensor[..., 3] / 2  #set xmax
#     tensor1[..., 3] = tensor[..., 1] + tensor[..., 2] / 2  #set ymax
#     tensor1[tensor1[..., 0] < 0] = 0; tensor1[tensor1[..., 0] > img_height] = img_height  # 0<xmin<img_heighit
#     tensor1[tensor1[..., 1] < 0] = 0; tensor1[tensor1[..., 1] > img_width] = img_width    # 0<ymin<img_width
#     tensor1[tensor1[..., 2] < 0] = 0; tensor1[tensor1[..., 2] > img_height] = img_height  # 0<xmax<img_heighit
#     tensor1[tensor1[..., 3] < 0] = 0; tensor1[tensor1[..., 3] > img_width] = img_width    # 0<ymax<img_width
#     tensor2[..., 0] = tensor1[..., 0] / img_height
#     tensor2[..., 1] = tensor1[..., 1] / img_width
#     tensor2[..., 2] = (tensor1[..., 3] - tensor1[..., 1]) / img_width
#     tensor2[..., 3] = (tensor1[..., 2] - tensor1[..., 0]) / img_height
#     return tensor2


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
#
#
# def ssd_box_encoder_one_image(roi_list, classes_list, num_classes,
#                                  anchors, iou_thresh_pos, iou_thresh_neg):
#     eps = 1e-6
#     (num_boxes, channels) = np.shape(anchors)
#     y_classses = np.zeros((1, num_boxes, num_classes + 1))  # '+1'代表背景类别
#     y_anchors = np.zeros((1, num_boxes, 4))
#     for id_anc, anchor in enumerate(anchors):
#         for roi_i, class_i in zip(roi_list, classes_list):
#             # x:anchor[0] y:anchor[1] w:anchor[2] h:anchor[3]
#             gt_x = roi_i[0]; gt_y = roi_i[1]; gt_w = roi_i[2]; gt_h = roi_i[3]
#             anchor_x = anchor[0]; anchor_y = anchor[1]; anchor_w = anchor[2]; anchor_h = anchor[3]
#             y_anchors[0, id_anc, 0] = (gt_x - anchor_x) / anchor_h
#             y_anchors[0, id_anc, 1] = (gt_y - anchor_y) / anchor_w
#             y_anchors[0, id_anc, 2] = np.log(gt_w / anchor_w)
#             y_anchors[0, id_anc, 3] = np.log(gt_h / anchor_h)
#             if iou_eval(roi_i, anchor) >= iou_thresh_pos:
#                 y_classses[0, id_anc, int(class_i)] = 1
#             elif iou_eval(roi_i, anchor) >= iou_thresh_neg and iou_eval(roi_i, anchor) < iou_thresh_pos:
#                 y_classses[0, id_anc, 0] = 1
#     return y_classses, y_anchors
#
#
# def ssd_box_encoder_batch(roi_list, classes_list, num_classes,
#                              anchors, iou_thresh_pos, iou_thresh_neg):
#       (batch, num_boxes, channels) = np.shape(anchors)
#       y_classses = np.zeros((batch, num_boxes, num_classes + 1))  # '+1'代表背景类别
#       y_anchors = np.zeros((batch, num_boxes, 4))
#       for i in range(batch):
#           y_classses[i, :, :], y_anchors[i, :, :] = ssd_box_encoder_one_image(
#                                                              roi_list=roi_list[i],
#                                                              classes_list=classes_list[i],
#                                                              num_classes=num_classes,
#                                                              anchors=anchors[i, :, :],
#                                                              iou_thresh_pos=iou_thresh_pos,
#                                                              iou_thresh_neg=iou_thresh_neg)
#       return y_classses, y_anchors


def rpn_box_encoder(obj, anchors, iou_pos_thresh, iou_neg_thresh):
    y_classes = np.zeros(2)   # +1代表背景
    classes_list = []; offset_list = []; name_list = []
    for anchor_i in anchors:
        iou_list = []
        for obj_i in obj:         #anchor_i与所有obj计算iou和offset，保存到iou_list列表中
            rect = obj_i['rect_corner']
            obj_name = obj_i['obj_name']
            iou = iou_eval(gt=rect, dr=anchor_i)
            offset_x = rect[0] - anchor_i[0]  #此处offset计算方式与原文不同
            offset_y = rect[1] - anchor_i[1]  #可以替换为指数形式编码
            offset_w = rect[2] - anchor_i[2]
            offset_h = rect[3] - anchor_i[3]
            offset = np.array([offset_x, offset_y, offset_w, offset_h])
            iou_list.append([iou, offset, obj_name])
        iou_list2 = sorted(iou_list, key=lambda x: x[0], reverse=True) #对iou_list排序，如果最大iou>thresh，就作为前景
        if iou_list2[0][0] >= iou_pos_thresh:  #标记正样本
            y_classes[1] = 1
        if iou_list2[0][0] >= iou_neg_thresh and iou_list2[0][0] <iou_pos_thresh: #标记hard sample，对于负样本不标记
            y_classes[0] = 1
        classes_list.append([y_classes])
        offset_list.append([iou_list2[0][1]])
        name_list.append(iou_list2[0][2])
    classes = np.reshape(np.array(classes_list), newshape=(-1, 2))
    offset = np.reshape(np.array(offset_list), newshape=(-1, 4))
    return classes, offset, name_list























if __name__ == '__main__':
    roi_list = [np.array([[0.1, 0.2, 0.1, 0.1]])]
    classes_list = [np.array([[1]])]
    anchors = np.zeros([1, 20646, 4])
    # y_classes, y_anchors = ssd_box_encoder_batch(roi_list=roi_list,
    #                                              classes_list=classes_list,
    #                                              anchors=anchors,
    #                                              iou_threshold=-0.5,
    #                                              num_classes=1)
    # print(np.shape(y_classes))
    # print(np.shape(y_anchors))






