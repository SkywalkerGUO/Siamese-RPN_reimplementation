import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from tqdm import tqdm
from network import SiamRPN
from config import config
from custom_transforms import ToTensor
from util_func import get_exemplar_image, get_instance_image, box_transform_inv,add_box_img,add_box_img_left_top,show_image,generate_anchors

class SiamRPNTracker():
    def __init__(self, model_path):
        self.name='SiamRPN'
        #load model
        self.model = SiamRPN()
        model_loaded = torch.load(model_path)
        if 'model' in model_loaded.keys():
            self.model.load_state_dict(torch.load(model_path)['model']).cuda()
        else:
            self.model.load_state_dict(torch.load(model_path)).cuda()
        #initialize eval
        self.model.eval()
        #initialize transformer
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales, config.anchor_ratios, config.valid_scope)
        #get cosine window
        self.score_outer = np.outer(np.hanning(config.score_size), np.hanning(config.score_size))
        self.window = np.tile(self.score_outer[None, :], [config.anchor_num, 1, 1])
        self.window = self.window.flatten()

    #first frame
    def init(self, frame, bbox):
        #[l,t,w,h]->[center_x,center_y,w,h]
        #center box
        center_total = [bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2 , bbox[2], bbox[3]]
        self.bbox = np.array(center_total)
        #get center point[c_x,c_y]
        center_point = [bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2]
        self.pos = np.array(center_point)
        #get target width[w,h]
        self.target_sz = np.array([bbox[2], bbox[3]])
        self.origin_target_sz = np.array([bbox[2], bbox[3]])
        #get mean of R/G/B 
        self.img_mean = np.mean(frame, axis=(0, 1))
        #exemplar_img
        exemplar_img, _, _ = get_exemplar_image(frame, self.bbox,config.exemplar_size, config.context_amount, self.img_mean)
        #add dimentions of batch
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]
        self.model.track_init(exemplar_img.cuda())

    def get_final_coordinates(self,target,frame,lr_box):
        #res_x,c_x:(0~w),res_y,c_y:(0~h)
        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])
        w_1 = self.target_sz[0] * (1 - lr_box) + target[2] * lr_box
        w_2 = config.min_scale * self.origin_target_sz[0], config.max_scale * self.origin_target_sz[0]
        h_1 = self.target_sz[1] * (1 - lr_box) + target[3] * lr_box
        h_2 = config.min_scale * self.origin_target_sz[1], config.max_scale * self.origin_target_sz[1]
        res_w = np.clip(w_1, w_2)
        res_h = np.clip(h_1, h_2)
        return res_x,res_y,res_w,res_h

    def update(self, frame):
        #input:bbox of last frame, keep img_mean
        get_instance_img, z_, y_, x_ = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.img_mean)
        #add batch dimension
        instance_img_trans = self.transforms(get_instance_img)[None, :, :, :].cuda()
        scores, regressions = self.model.track_update(instance_img_trans)
        dim_3 = config.anchor_num * config.score_size * config.score_size
        #[1,10,19,19]->[1,2,5*19*19]->[1,1805,2]
        confs = scores.reshape(-1, 2, dim_3).permute(0,2,1)
        #[1,20,19,19]->[1,4,5*19*19]->[1,1805,4]
        offsets = regressions.reshape(-1, 4, dim_3).permute(0,2,1)
        delta = offsets[0].cpu().detach().numpy()
        #get anchor
        box_pred = box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(confs, dim=2)[0, :, 1].cpu().detach().numpy()

        #size
        box_pad1 = (box_pred[:, 2] + box_pred[:, 3]) * 0.5
        get_sz1 = (box_pred[:, 2] + box_pad1) * (box_pred[:, 3] + box_pad1)
        sz1 = np.sqrt(get_sz1)

        target_wh = self.target_sz * x_
        target_pad2 = (target_wh[0] + target_wh[1]) * 0.5
        target_sz2 = (target_wh[0] + target_pad2) * (target_wh[1] + target_pad2)
        sz2 = np.sqrt(target_sz2)

        sz_f = sz1/sz2
        s_c = np.maximum(sz_f, 1. / sz_f)
        #ratio
        ratio1 = self.target_sz[0] / self.target_sz[1]
        ratio2 = box_pred[:, 2] / box_pred[:, 3]
        ratio_f = ratio1/ratio2
        r_c = np.maximum(ratio_f, 1. / ratio_f)
    
        # combine size and ratio penalty
        penalty_exp = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        #get penalty socres
        penalty_score = penalty_exp * score_pred#对每一个anchors的正样本分类预测分数×惩罚因子
        #calculate cosine window
        cosine_window = (1 - config.window_influence) + self.window * config.window_influence
        penalty_score = penalty_score*cosine_window
        best_score_index = np.argmax(penalty_score) #返回最大得分的索引id
        
        target = box_pred[best_score_index, :] / x_
        lr_box = penalty_exp[best_score_index] * score_pred[best_score_index] * config.lr_box
        
        res_x,res_y,res_w,res_h = self.get_final_coordinates(target,frame,lr_box)
        #print(res_x)
        
        #get new center, w, h
        bbox = np.array([res_x, res_y, res_w, res_h])
        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])

        #target->image
        self.bbox = (
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64))

        #[c_x,c_y,w,h]->[l,t,w,h]
        bbox_dim1 = self.pos[0] + 1 - (self.target_sz[0]-1) / 2
        bbox_dim2 = self.pos[1] + 1 - (self.target_sz[1]-1) / 2
        bbox_dim3 = self.target_sz[0]
        bbox_dim4 = self.target_sz[1]

        bbox=np.array([bbox_dim1,bbox_dim2,bbox_dim3,bbox_dim4])

        return bbox

    #test for OTB
    def track(self, img, box, visualize=False):
        frame_num = len(img)
        times = np.zeros(frame_num)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        for i, img_one in enumerate(img):
            img_one_ = cv2.imread(img_one, cv2.IMREAD_COLOR)
            img_one_ = cv2.cvtColor(img_one_,cv2.COLOR_BGR2RGB)
            begin = time.time()
            if i == 0:
                self.init(img_one_, boxes[0])
            else:
                boxes[i, :] = self.update(img_one_)
            times[i] = time.time() - begin
            if visualize:
                show_image(img_one_, boxes[i, :])
        return boxes, times

    #get cosine window
    def cosine_window(self, size):
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :]).astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window