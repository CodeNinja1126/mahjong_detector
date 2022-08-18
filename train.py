import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT

from tqdm import tqdm
from torch.utils.data import DataLoader
from darknet import Darknet

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64 # 64 in original paper
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'yolov3.pt'
IMG_DIR = 'data/images'
LABEL_DIR = 'data/labels'
CFGS_DIR = 'cfg/yolov3.cfg'

def main():
    # load pretrained model
    print('Loading network.....')
    model = Darknet(CFGS_DIR)
    # model.load_state_dict(torch.load(LOAD_MODEL_FILE))
    model.load_weights(LOAD_MODEL_FILE)
    print('Network successfully loaded')


    '''≈
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    '''
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
    )

    '''
    loading model code
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    '''

    '''
    data_set and data_loader code
    train_dataset = VOCDataset(
        "data/train.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    test_dataset = VOCDataset(
        'data/test.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR

    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    '''


if __name__ == '__main__':
    main()