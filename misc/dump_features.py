import torch
import torchvision.transforms as T

import os
import time
from tqdm import tqdm
import joblib

import argparse
import numpy as np

import utils.bagnet
from utils.dataset import *


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help='model directory')
parser.add_argument("--data_dir",default='data',type=str,help='data directory')
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--model",default='bagnet33',type=str,help='model name')

args = parser.parse_args()

DATASET = args.dataset
MODEL_DIR = args.model_dir
DATA_DIR = os.path.join(args.data_dir,DATASET)
NUM_CLS = NUM_CLASS_DICT[DATASET]


# set up dataset
if DATASET == 'voc':
	transforms = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	val_dataset = VOCDetection(DATA_DIR,image_set='test',year='2007',transforms=transforms)
elif DATASET == 'coco':
	val_dataset = get_coco(DATA_DIR,image_set='val')
elif DATASET == 'kitti':
	val_dataset = Kitti2DDetection(DATA_DIR,image_set='val')
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False,pin_memory=True,collate_fn=collate_fn,num_workers=2)


# build BagNet used in Objectness Preidctor
if 'bagnet33' in args.model:
	bagnet_model = utils.bagnet.bagnet33(aggregation='none')
	FM_SIZE = FM_SIZE_DICT[DATASET] #hard-coded feature map size for now
	RF_SIZE = 33

device='cuda'
num_ftrs = bagnet_model.fc.in_features
bagnet_model.fc = torch.nn.Linear(num_ftrs, NUM_CLS+1)
bagnet_model = torch.nn.DataParallel(bagnet_model)
checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_{}.pth'.format(DATASET)))
bagnet_model = bagnet_model.to(device)
bagnet_model.load_state_dict(checkpoint['model_state_dict'])
bagnet_model.eval()


feature_list =[]
target_list = []

for img,targets in tqdm(val_loader):

	assert len(targets[0]['labels'])>0

	local_features = bagnet_model(img.cuda()).detach().cpu().numpy().squeeze().transpose(1,2,0)
	local_features = np.clip(local_features,0,np.inf)
	feature_list.append(local_features)
	target_list.append(targets)

feature_list = np.stack(feature_list)
joblib.dump(feature_list,os.path.join(DATA_DIR,'feature_list_{}_{}.z'.format(args.model,DATASET)))
joblib.dump(target_list,os.path.join(DATA_DIR,'target_list_{}_{}.z'.format(args.model,DATASET)))
