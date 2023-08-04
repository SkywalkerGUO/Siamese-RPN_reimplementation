import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

#calculate cross-entropy and smoothL1
def intersection_over_union(boxes1, boxes2):
    #calculate iou
    M = boxes1.shape[0]
    N = boxes2.shape[0]
    
    x1 = boxes1[:,0].unsqueeze(-1).repeat(1,N)
    x2 = boxes1[:,1].unsqueeze(-1).repeat(1,N)
    x3 = boxes1[:,2].unsqueeze(-1).repeat(1,N)
    x4 = boxes1[:,3].unsqueeze(-1).repeat(1,N)

    y1 = boxes2[:,0].repeat(M,1)
    y2 = boxes2[:,1].repeat(M,1)
    y3 = boxes2[:,2].repeat(M,1)
    y4 = boxes2[:,3].repeat(M,1)

    area_box1 = torch.mul(x3-x1, x4-x2)
    area_box2 = torch.mul(y3-y1, y4-y2) 

    max_x1 = torch.max(x1,y1)
    max_x2 = torch.max(x2,y2)
    min_x3 = torch.min(x3,y3)
    min_x4 = torch.min(x4,y4)
    x1_intersect = torch.clamp(max_x1,min=0)
    x2_intersect = torch.clamp(max_x2,min=0)
    x3_intersect = torch.clamp(min_x3,min=0)
    x4_intersect = torch.clamp(min_x4,min=0)

    intersect_area = (x3_intersect - x1_intersect).clamp(min=0.0) * (x4_intersect - x2_intersect).clamp(min=0.0)
    union_area = (area_box2 - intersect_area) + area_box1
    iou = intersect_area/union_area
    
    return iou

def nms(boxes, scores, num, threshold=0.7):
    sorted = np.argsort(scores)[::-1] 
    sort_boxes = boxes[sorted]
    box_get = [sort_boxes[0]]
    index_get = [sorted[0]]
    for i, box in enumerate(sort_boxes):
        #calculate iou
        iou = intersection_over_union(box_get, box)
        max_iou = np.max(iou)
        #compare the max_iou with threshold
        if max_iou < threshold:
            box_get.append(box)
            index_get.append(sorted[i])
            #fill the box
            if len(box_get) == num:
                break
        else:
            pass
    return index_get

def generate_anchors(stride, batch_size, scales, ratios, score):
    #get anchor of different sizes
    anchor_total = len(ratios) * len(scales)
    count = 0
    col_size = 4
    anchor = np.zeros((anchor_total, col_size), dtype=np.float32) 
    area = batch_size**2
    
    #get anchor of width and height
    for each_ratio in ratios:
        width = int(np.sqrt(area/each_ratio))
        height = int(width*each_ratio)
        for each_scale in scales:
            anchor_width = width * each_scale
            anchor_height = height * each_scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = anchor_width
            anchor[count, 3] = anchor_height
            count += 1
    
    get_tile_scores = score**2
    #make anchor col_size=4 
    anchor= np.tile(anchor, get_tile_scores).reshape((-1, col_size))
    #get anchor from center
    each_size = score // 2
    center_point =-each_size*stride
    #get anchor of feature map
    xx_line_set = []
    yy_line_set = []
    #get point set xx and yy

    for i in range(score):
        x_point = center_point + stride * i
        y_point = center_point + stride * i
        xx_line_set.append(x_point)
        yy_line_set.append(y_point)

    #get flatten xx yy set 
    xx = np.tile(xx_line_set.flatten(), (anchor_total, 1)).flatten()
    yy = np.tile(yy_line_set.flatten(), (anchor_total, 1)).flatten()
    anchor[:, 0] = xx.astype(np.float32)
    anchor[:, 1] = yy.astype(np.float32)
    return anchor