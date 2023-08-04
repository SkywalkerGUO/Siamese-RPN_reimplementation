import torch
import torch.nn
import torch.nn.functional as F
import random
import numpy as np
from anchor import intersection_over_union,nms,generate_anchors

def rpn_cross_entropy(input, target, num_pos, num_neg, anchors, reg_pos=None, reg_neg=None):
    loss_res = []
    M = target.shape[0]
    for image_one in range(M):
        pos_index = np.where(target[image_one].cpu() == 1)[0]
        min_pos = min(len(pos_index), num_pos) 
        num_index = len(pos_index) * num_neg / num_pos
        min_neg = min(num_index, num_neg)
        min_neg = int(min_neg)
        #the positive index of target
        first_index = np.where(target[image_one].cpu() == 1)[0]
        #the negative index of target
        second_index = np.where(target[image_one].cpu() == 0)[0]
        pos_index = first_index.tolist()
        neg_index = second_index.tolist()

        if reg_pos:
            #use nms on anchor
            pos_num = len(pos_index)
            if pos_num > 0:
                #calculate cross entropy loss
                pos_loss_one = F.cross_entropy(input=input[image_one][pos_index],target=target[image_one][pos_index], reduction='none')
                #perform nms
                pos_loss_final = pos_loss_one[nms(anchors[pos_index], pos_loss_one.cpu().detach().numpy(), min_pos)]
            else:
                if torch.cuda.is_available():
                    pos_loss_final = torch.FloatTensor([0]).cuda()
                else:
                    pos_loss_final = torch.FloatTensor([0])
        else:
            pos_random = random.sample(pos_index, min_pos)
            pos_num = len(pos_index)
            if pos_num > 0:
                #calculate cross entropy loss
                pos_loss_final = F.cross_entropy(input=input[image_one][pos_random],target=target[image_one][pos_random], reduction='none')
            else:
                if torch.cuda.is_available():
                    pos_loss_final = torch.FloatTensor([0]).cuda()
                else:
                    pos_loss_final = torch.FloatTensor([0])
        if reg_neg:
            #use nms on anchor
            pos_num = len(pos_index)
            #calculate cross entropy loss
            neg_loss_one = F.cross_entropy(input=input[image_one][neg_index],target=target[image_one][neg_index], reduction='none')
            #perform nms
            if pos_num > 0:
                neg_loss_final = neg_loss_one[nms(anchors[neg_index], neg_loss_one.cpu().detach().numpy(), min_neg)]
            else:
                neg_loss_final = neg_loss_one[nms(anchors[neg_index], neg_loss_one.cpu().detach().numpy(), num_neg)]
        else:
            pos_num = len(pos_index)
            if pos_num > 0:
                #sample
                neg_random = random.sample(neg_index, min_neg)
                #calculate cross entropy loss
                neg_loss_final = F.cross_entropy(input=input[image_one][neg_random],target=target[image_one][neg_random], reduction='none')
            else:
                pos_where = np.where(target[image_one].cpu() == 0)[0]
                neg_random = random.sample(pos_where.tolist(), num_neg)
                #calculate cross entropy loss
                neg_loss_final = F.cross_entropy(input=input[image_one][neg_random],target=target[image_one][neg_random], reduction='none')
        #calculte mean of the loss
        pos_loss_final_mean = pos_loss_final.mean()
        neg_loss_final_mean = neg_loss_final.mean()
        loss = (pos_loss_final_mean + neg_loss_final_mean) / 2
        loss_res.append(loss)
    final_loss_result = torch.stack(loss_res).mean()
    return final_loss_result


def rpn_smoothL1(input, target, label, num_pos=16, reg=None):
    #implement smoothL1 loss function
    loss_all = []
    M = target.shape[0]
    for image_one in range(M):
        #get index
        pos_index = np.where(label[image_one].cpu() == 1)[0]
        min_pos = min(len(pos_index), num_pos)
        if reg:
            pos_index = np.where(label[image_one].cpu() == 1)[0]
            if len(pos_index) > 0:
                #calculate the smooth_L1 loss
                loss_L1 = F.smooth_l1_loss(input[image_one][pos_index], target[image_one][pos_index], reduction='none')
                loss_sort_index = torch.argsort(loss_L1.mean(1))
                loss_reg = loss_L1[loss_sort_index[-num_pos:]]
            else:
                loss_reg = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_reg.mean())
        else:
            pos_index = np.where(label[image_one].cpu() == 1)[0]
            pos_sample = random.sample(pos_index.tolist(), min_pos)
            if len(pos_sample) > 0:
                #calculate the smooth_L1 loss
                loss_L1 = F.smooth_l1_loss(input[image_one][pos_sample], target[image_one][pos_sample])
            else:
                loss_L1 = torch.FloatTensor([0]).cuda()[0]
            loss_all.append(loss_L1.mean())
    result = torch.stack(loss_all).mean()
    return result