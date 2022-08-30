import torch.nn as nn
import torch.nn.functional as F
from util import *


class yolo_loss(nn.Module):
    def __init__(self, 
                 anchors,
                 grid_sizes,
                 inp_dim,
                 CUDA,
                 confidence=0.5,
                 lambda_coord = 1.0,
                 lambda_conf_obj = 5.0,
                 lambda_conf_noobj = 0.5):
        super(yolo_loss,self).__init__()
        # 앵커 사이즈((10, 13), ...)
        self.anchors = torch.tensor(anchors)
        # 그리드 사이즈 (13, ...)
        self.grid_sizes = torch.tensor(grid_sizes)
        if CUDA:
            self.anchors = self.anchors.cuda()
            self.grid_sizes = self.grid_sizes.cuda()
        self.inp_dim = inp_dim
        self.confidence = confidence
        self.lambda_coord = lambda_coord
        self.lambda_conf_obj = lambda_conf_obj
        self.lambda_conf_noobj = lambda_conf_noobj
        self.CUDA = CUDA
    

    def forward(self, pred_x, coord_x, y_cls, y_coord):
        
        box_x = pred_x.new(pred_x.shape)
        box_x[:,:,0] = (pred_x[:,:,0] - pred_x[:,:,2]/2)
        box_x[:,:,1] = (pred_x[:,:,1] - pred_x[:,:,3]/2)
        box_x[:,:,2] = (pred_x[:,:,0] + pred_x[:,:,2]/2)
        box_x[:,:,3] = (pred_x[:,:,1] + pred_x[:,:,3]/2)
        pred_x[:,:,:4] = box_x[:,:,:4]
        
        # coord_loss
        idx_obj = [[],[]]
        target_coord = []
        target_class = []
        for i in range(len(pred_x)):
            for box in y_coord[i]:
                # 타겟의 중심이 들어가는 셀들을 골라내 해당 셀의 바운딩 박스만을 비교해야 됨.
                # 해당 셀들의 바운딩 박스를 골라내는 코드
                candis = []
                base_idx = 0
                for g in self.grid_sizes:
                    tmp_idx = base_idx + ((box[1] * g // 1.0) + (box[0] * g // 1.0) * g) * len(self.anchors[0])
                    tmp_idx = int(tmp_idx)
                    candis += list(range(tmp_idx, tmp_idx+len(self.anchors[0])))
                    base_idx += g ** 2 * 3

                tmp_box = box.new(box.shape)
                tmp_box[0] = self.inp_dim * (box[0] - box[2]/2)
                tmp_box[1] = self.inp_dim * (box[1] - box[3]/2)
                tmp_box[2] = self.inp_dim * (box[0] + box[2]/2)
                tmp_box[3] = self.inp_dim * (box[1] + box[3]/2)

                # iou가 가장 큰 박스를 구해냄
                ious = bbox_iou(tmp_box.unsqueeze(0), pred_x[i, candis])
                train_idx = ious.argmax()

                # 해당 박스의 인덱스를 저장
                idx_obj[0].append(i) 
                idx_obj[1].append(candis[train_idx])

                # 데이터 레이블을 학습할 수 있게 변환

                grid_idx = train_idx//len(self.anchors)
                anchor_idx = train_idx%len(self.anchors)
                anchor_size = self.anchors[grid_idx][anchor_idx]

                tmp_box[0] = (box[0] * self.grid_sizes[grid_idx]) % 1.0
                tmp_box[1] = (box[1] * self.grid_sizes[grid_idx]) % 1.0

                inverse_logistic = lambda x:torch.log(x/(1-x))
                # 1e-5는 eps
                tmp_box[:2] = inverse_logistic(tmp_box[:2]+1e-5)
                
                tmp_box[2:] = box[2:] * self.inp_dim
                tmp_box[2:] = torch.log(tmp_box[2:]/anchor_size)

                target_coord.append(tmp_box.unsqueeze(0))
            
            # 클래스 레이블
            for cls in y_cls[i]:
                target_class.append(cls)
        
        
        target_coord = torch.cat(target_coord, 0)
        target_class = torch.tensor(target_class)
        target_class = F.one_hot(target_class, num_classes=len(pred_x[0,0,5:])).float()
        if self.CUDA:
            target_class = target_class.cuda()
        
        target_confidence = torch.zeros(pred_x[:,:,4].shape)
        if self.CUDA:
            target_confidence = target_confidence.cuda()
        target_confidence[idx_obj] = 1



        coord_loss = self.lambda_coord * F.mse_loss(coord_x[idx_obj], target_coord, reduction='sum')
        confidence_loss = self.lambda_conf_obj * \
                          F.binary_cross_entropy(pred_x[:,:,4][target_confidence==1] ,
                                                 target_confidence[target_confidence==1], reduction='sum') + \
                          self.lambda_conf_noobj * \
                          F.binary_cross_entropy(pred_x[:,:,4][target_confidence==0],
                                                 target_confidence[target_confidence==0], reduction='sum')
        classifier_loss = F.binary_cross_entropy(pred_x[idx_obj][:,5:], target_class, reduction='sum')

        return coord_loss + confidence_loss + classifier_loss