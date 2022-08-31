import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util import *
from data import Yolo_v3_dataset
from loss import yolo_loss
from darknet import Darknet


seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 5e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32 # 64 in original paper
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 100
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'yolov3.pt'
# IMG_DIR = 'data/images'
# LABEL_DIR = 'data/labels'
# CFGS_DIR = 'cfg/yolov3.cfg'

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Train Module')

    parser.add_argument('--data', 
                        dest='data',
                        help='data directory', 
                        type=str)
    parser.add_argument('--bs', 
                        dest='bs',
                        help='Batch size',
                        default=32, 
                        type=int)
    parser.add_argument('--confidence', 
                        dest='confidence',
                        help='Object Confidence to filter predictions',
                        default=0.5, 
                        type=float)
    parser.add_argument('--nms_thresh', 
                        dest='nms_thresh',
                        help='NMS Threshold',
                        default=0.4, 
                        type=float)
    parser.add_argument('--cfg', 
                        dest='cfgfile',
                        help='Config file',
                        default='cfg/yolov3_mahjong.cfg', 
                        type=str)
    parser.add_argument('--weights', 
                        dest='weightsfile',
                        help='weightsfile',
                        default='yolov3.pt', 
                        type=str)
    parser.add_argument('--reso', 
                        dest='reso',
                        help='Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
                        default='416', 
                        type=str)
    parser.add_argument('--classes',
                        dest='cls_file',
                        help='Classes file',
                        default='data/coco.names',
                        type=str)
    parser.add_argument('--epoch',
                        dest='epoch',
                        help='num epochs',
                        default=100,
                        type=int)
    parser.add_argument('--lr',
                        dest='lr',
                        help='learning_rate',
                        default=5e-6,
                        type=float)

    return parser.parse_args()



def main():
    args = arg_parse()
    CUDA = torch.cuda.is_available()
    
    # load pretrained model
    print('Loading network.....')
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print('Network successfully loaded')

    # read grid sizes and anchors
    grid_sizes = tuple(map(float, model.net_info['grid_sizes'].split(',')))
    grid_sizes = torch.tensor(grid_sizes)
    
    anchors = tuple(map(float, model.net_info['anchors'].split(',')))
    anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
    anchors = [(anchors[i], anchors[i+1], anchors[i+2]) for i in range(0,len(anchors),3)]
    anchors = torch.tensor(anchors)
    
    inp_dim = int(model.net_info['height'])

    if CUDA:
        model = model.cuda()
        anchors = anchors.cuda()
        grid_sizes = grid_sizes.cuda()

    # loss
    criterion = yolo_loss(anchors,
                          grid_sizes,
                          inp_dim,
                          CUDA=CUDA)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

    train_dataset = Yolo_v3_dataset(args.data, inp_dim)
    data_loader = DataLoader(train_dataset, 
                             batch_size=args.bs, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True, 
                             drop_last=True
                            )
    
    model.train()

    running_loss = 0.0
    for epoch in range(args.epoch):
        for batch in data_loader:
            inp, idx = batch

            cls_label = []
            coord_label = []
            for i in idx:
                label = train_dataset.get_label(i)
                cls_label.append([a[0] for a in label])
                tmp_coord = [torch.tensor(a[1]).float() for a in label]
                if CUDA:
                    tmp_coord = [a.cuda() for a in tmp_coord]
                coord_label.append(tmp_coord)
            
            x = inp.squeeze(1)
            if CUDA:
                x = x.cuda()
            
            optimizer.zero_grad()

            x = model(x, CUDA)
            loss = criterion(x[0], x[1], cls_label, coord_label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 9:    # print every 10 epoch
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 30:.3f}')
            running_loss = 0.0
            
        torch.cuda.empty_cache()

    ckpt = model.state_dict()
    torch.save(ckpt, 'yolov3_mahjong.pt')

if __name__ == '__main__':
    main()