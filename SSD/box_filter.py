import numpy as np

def box_filter(pred_offset, pred_classes, pred_anchors, num_positives):
    pred_rect_list = []
    (batch_size, num_boxes, num_classes) = np.shape(pred_classes)
    merge = np.concatenate((pred_classes, pred_offset, pred_anchors), axis=2)  #ndarray [batch_size, num_boxex, num_class+4+4]
    for i in range(batch_size):
       merge_i = np.lexsort(keys=(merge[i, :, 1], merge[i, :, :]), axis=0)
       top_k_merge = merge_i[-num_positives:-1, :]
       top_k_merge[:, 6] = top_k_merge[:, 6] + top_k_merge[:, 2]      #解码过程
       top_k_merge[:, 7] = top_k_merge[:, 7] + top_k_merge[:, 3]
       top_k_merge[:, 8] = top_k_merge[:, 8] * np.exp(top_k_merge[:, 4])
       top_k_merge[:, 9] = top_k_merge[:, 9] * np.exp(top_k_merge[:, 5])
       rect_pred = np.mean(a=top_k_merge, axis=0)
       pred_rect_list.append(rect_pred)
    return np.array(pred_rect_list)










    return


def class_pred_acc(cls_pred, cls_true):
    (batch_size, boxes, classes) = np.shape(cls_pred)
    num_bk = 0; num_correct_bk = 0; num_cls = 0; num_correct_cls = 0; num_bk_to_cls = 0
    index_pred = np.argmax(a=cls_pred, axis=2)
    index_trued = np.argmax(a=cls_true, axis=2)
    index_pred = np.reshape(a=index_pred, newshape=(batch_size * boxes))
    index_trued = np.reshape(a=index_trued, newshape=(batch_size * boxes))
    assert(len(index_pred) == len(index_trued), 'index_pred and index_trued are not equal')
    for pred_i, true_i in zip(index_pred, index_trued):
        if true_i == 0:
            num_bk += 1
            if pred_i == 0:
                num_correct_bk += 1
            if pred_i == 1:
                num_bk_to_cls += 1
        if true_i == 1:
            num_cls += 1
            if pred_i == 1:
                num_correct_cls += 1
    return num_correct_bk/num_bk, num_correct_cls/num_cls, np.sum(index_pred), np.sum(index_trued), \
           num_bk_to_cls/(num_correct_cls + num_bk_to_cls)




