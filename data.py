from __future__ import division
from torch.utils.data import Dataset
from util import *
import os
import os.path as osp
import cv2
import numpy as np

def get_dataset(dir, bs):
    pass


def transform(img_path, inp_dim):
    image = cv2.imread(img_path)
    return prep_image(image, inp_dim)


def target_transform(label):
    with open(label, 'r') as f:
        labels = f.read().split('\n')

    ret = []    
    for label in labels[:-1]:
        label = list(map(float, label.split()))
        class_idx = int(label[0])
        coord_label = np.array(label[1:])
        ret.append((class_idx, coord_label))
    
    return ret


class Yolo_v3_dataset(Dataset):
    def __init__(self, 
                 data_dir,
                 inp_dim,
                 transform=transform, 
                 target_transform=target_transform):
        self.img_dir = osp.join(data_dir, 'images/')
        self.label_dir = osp.join(data_dir, 'labelling/')
        self.inp_dim = inp_dim
        try:
            self.img_names = [name for name in os.listdir(self.img_dir) if name[-4:] == 'jpeg']
        except NotADirectoryError:
            print(f'Incorrect data dir.')
            exit()
            
        self.img_names.sort()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = osp.join(self.img_dir, self.img_names[idx])
        image = self.transform(img_path, self.inp_dim)
        return (image, idx)

    def get_label(self, idx):
        label = osp.join(self.label_dir, self.img_names[idx][:-4]+'txt')
        label = self.target_transform(label)
        return label


if __name__ == '__main__':
    data = Yolo_v3_dataset('/Users/heonoo/workspace/data_dir/data', 416)

    from torch.utils.data import DataLoader
    import torch
    data_loader = DataLoader(data, 
                             batch_size=32, 
                             shuffle=True, 
                             num_workers=4, 
                             pin_memory=True, 
                             drop_last=True
                            )
    for batch in data_loader:
        inp, idx = batch
        cls_label = []
        coord_label = []
        for i in idx:
            label = data.get_label(i)
            cls_label.append([a[0] for a in label])
            coord_label.append([torch.tensor(a[1]) for a in label])
        print(cls_label)
        print(coord_label)