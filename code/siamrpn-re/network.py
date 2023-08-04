import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from .custom_transforms import ToTensor

from torchvision.models import alexnet
from torch.autograd import Variable
from torch import nn

from .config import config


class SiameseAlexNet(nn.Module):
    def __init__(self,):
        super(SiameseAlexNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),  
            nn.BatchNorm2d(96),              
            nn.MaxPool2d(3, stride=2),      
            nn.ReLU(inplace=True), 

            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),      
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),             
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),             
        )
        self.anchor_num = config.anchor_num  
        self.input_size = config.instance_size
        self.size_sub = self.input_size - config.exemplar_size
        self.score_rep = int(self.size_sub / config.total_stride)
        self.examplar_cls = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.examplar_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)

        self.ins_cls = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.ins_r1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        self.regress = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)
        
    def get_pred_score(self,template,detection):
        #get prediction score
        N = template.size(0)
        regulizer = 2
        template_res = self.layers(template)
        detection_res = self.layers(detection)
        con_first_score = self.ins_cls(detection_res)
        conv_scores = con_first_score.reshape(1, -1, self.score_rep + 4, self.score_rep + 4)
        
        new_dim3 = self.score_rep + 1
        new_dim4 = new_dim3

        kernel_one_score = self.examplar_cls(template_res)
        kernel_one_score = kernel_one_score.view(N, regulizer*self.anchor_num, 256, 4, 4)

        score_filters = kernel_one_score.reshape(-1, 256, 4, 4)
        score = F.conv2d(conv_scores, score_filters, groups=N)
        pred_score_final = score.reshape(N, 10, new_dim3,new_dim4)
        return pred_score_final

    def get_pred_regression(self,template,detection):
        #get prediction regression
        N = template.size(0)
        regulizer = 4
        template_res = self.layers(template)
        detection_res = self.layers(detection)

        new_dim3 = self.score_rep + 1
        new_dim4 = new_dim3
        
        tem_regression = self.ins_r1(detection_res)
        tem_reg = tem_regression.reshape(1, -1, self.score_rep + 4, self.score_rep + 4)
        kernel_one_regression = self.examplar_r1(template_res).view(N, regulizer*self.anchor_num, 256, 4, 4)
        tem_fil = kernel_one_regression.reshape(-1, 256, 4, 4)
        pred_regression_final = self.regress(F.conv2d(tem_reg, tem_fil, groups=N))
        pred_regression_final = pred_regression_final.reshape(N, 20, new_dim3,new_dim4)

        return pred_regression_final

    def forward(self, template, detection):
        
        pred_score = self.get_pred_score(template,detection)
        
        pred_regression = self.get_pred_regression(template,detection)

        return pred_score, pred_regression

    def track_init(self, template):
        N = template.size(0)
        regular1 = 2
        regular2 = 4
        #output = [1, 256, 6, 6]
        template_res = self.layers(template)
        # kernel_score=1,2x5,256,4,4   kernel_regression=1,4x5, 256,4,4
        res_score = self.examplar_cls(template_res)
        res_score_final = res_score.view(N, regular1 * self.anchor_num, 256, 4, 4)

        res_regression = self.examplar_r1(template_res)
        res_regression_final = res_regression.view(N, regular2*self.anchor_num, 256, 4, 4)
        self.score_res = res_score_final.reshape(-1, 256, 4, 4)   # 2x5, 256, 4, 4  
        self.reg_res = res_regression_final.reshape(-1, 256, 4, 4)# 4x5, 256, 4, 4

    def track(self, detection):
        N = detection.size(0)
        #dim = [1,256,22,22]
        detection_res = self.layers(detection)
        
        score = self.ins_cls(detection_res)
        #print(score)
        regression = self.ins_r1(detection_res)
        #print(regression)
        new_dim1 = self.score_rep + 4
        new_dim2 = new_dim1
        scores_final = score.reshape(1, -1, new_dim1, new_dim2)
        new_dim3 = self.score_rep + 1
        new_dim4 = new_dim3

        pred_score = F.conv2d(scores_final, self.score_res, groups=N).reshape(N, 10, new_dim3, new_dim4)
        pred_score_final = pred_score.reshape(N, 10, new_dim3, new_dim4)

        reg_score = regression.reshape(1, -1, new_dim1, new_dim2)
        pred_regression = self.regress(F.conv2d(reg_score, self.reg_res, groups=N))
        pred_regression_final = pred_regression.reshape(N, 20, new_dim3, new_dim4)
        return pred_score_final, pred_regression_final