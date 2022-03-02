import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as T

from tqdm import tqdm
import time
import os
import copy

import argparse
import numpy as np
import math
import random

import nets.bagnet
from utils.defense import bboxes_img2fm
from coco_utils import get_coco,collate_fn,IMG_SIZE


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str)
parser.add_argument("--data_dir",default='data_coco',type=str)
parser.add_argument("--model_name",default='bagnet33_voc.pth',type=str)
parser.add_argument("--clip",default=-1,type=int)
parser.add_argument("--epoch",default=20,type=int)
parser.add_argument("--aggr",default='none',type=str)
parser.add_argument("--lr",default=0.001,type=float)
parser.add_argument("--resume",action='store_true')
parser.add_argument("--fc",action='store_true',help="only retrain the fully-connected layer")
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

train_dataset = get_coco(DATA_DIR,image_set='train')
val_dataset = get_coco(DATA_DIR,image_set='val')

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,shuffle=True,pin_memory=True,collate_fn=collate_fn,num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16,shuffle=False,pin_memory=True,collate_fn=collate_fn,num_workers=2)

NUM_CLS = 80
dataloaders={'train':train_loader,'val':val_loader}

device = 'cuda'#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_sizes = {'train':len(train_dataset),'val':len(val_dataset)}


def train_model(model, criterion, optimizer, scheduler, num_epochs=20):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            running_loss_neg = 0.0
            running_corrects_neg = 0
            running_total_neg = 0
            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs.size())
                    loss_pos,loss_neg,corr,total,corr_neg,total_neg = criterion(outputs, targets)
                    loss = loss_pos + 1*loss_neg
                    #print(loss_pos,loss_neg)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_pos.item() * inputs.size(0)
                running_corrects += corr
                running_total += total
                running_loss_neg += loss_neg.item() * inputs.size(0)
                running_corrects_neg += corr_neg
                running_total_neg += total_neg

            if phase == 'train':
                scheduler.step()
           
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / running_total
            epoch_loss_neg = running_loss_neg / dataset_sizes[phase]
            epoch_acc_neg = running_corrects_neg / running_total_neg            
            print('{} Loss: {:.4f} Acc: {:.4f}  Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc,epoch_loss_neg,epoch_acc_neg))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saving...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()
                    }, os.path.join(MODEL_DIR,args.model_name))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None


if 'bagnet17' in args.model_name:
    model_conv = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    FM_SIZE = 50 #hard-code the feature map size for now 
    RF_SIZE = 17
elif 'bagnet33' in args.model_name:
    model_conv = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    FM_SIZE = 48
    RF_SIZE = 33
elif 'resnet50' in args.model_name:
    model_conv = nets.resnet.resnet50(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    FM_SIZE = 13#48 for bagnet33 50 for bagnet17#13for resnet50


#FM_IMG_RATIO = FM_SIZE/IMG_SIZE

if args.fc: #only retrain the fully-connected layer
	for param in model_conv.parameters():
	    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, NUM_CLS+1)

model_conv = torch.nn.DataParallel(model_conv)

#remove border for now

def bboxes_img2fm(img_bboxes,FM_SIZE,RF_SIZE,stride=8):
    img_bboxes = img_bboxes.copy()
    img_bboxes[:,:2] = img_bboxes[:,:2] - RF_SIZE/2
    img_bboxes[:,2:] = img_bboxes[:,2:] - RF_SIZE/2
    fm_bboxes = img_bboxes / 8
    fm_bboxes = np.ceil(fm_bboxes)
    fm_bboxes = fm_bboxes[:,[1,0,3,2]]
    fm_bboxes = np.clip(fm_bboxes,0,FM_SIZE)


    return fm_bboxes.astype(int)

neg_label = torch.tensor([NUM_CLS]).cuda() # for background
def criterion(logits_list, targets):
    B,C,W,H = logits_list.size()
    loss_pos = 0
    loss_neg = 0
    corr_pos = 0
    total_neg = 0
    corr_neg = 0
    total_pos = 0
    for i in range(B):
        tar = targets[i]
        boxes = tar['bbox']
        if len(boxes) > 0 :
            mask  = torch.ones(size=(FM_SIZE,FM_SIZE),dtype=bool).cuda()
            pad = [int(x//8) for x in tar['pad']]
            if pad[1]>0:
                pad[1]+=1
                mask[:pad[1]]=False
                mask[-pad[1]:]=False
            elif pad[0]>0:
                pad[0]+=1
                mask[:,:pad[0]]=False
                mask[:,-pad[0]:]=False
            #mask = mask.cuda()
            boxes = bboxes_img2fm(boxes,IMG_SIZE,FM_SIZE,RF_SIZE)#.astype(int)
            boxes[:,:2]+=1
            boxes[:,2:]-=1
            boxes = np.clip(boxes,0,FM_SIZE)
            ########################################################################
            logits = torch.where(mask,torch.clamp(logits_list[i],0,torch.tensor(float('inf'))),torch.tensor(0.).cuda())
            #logits = torch.clamp(logits_list[i],0,torch.tensor(float('inf')))

            labels = tar['labels'].cuda()
            reduced_logits = []
            mask2 = mask.clone()
            
            for j,xyxy in enumerate(boxes):
                region = logits[:,xyxy[0]:xyxy[2],xyxy[1]:xyxy[3]]
                C,W,H = region.size()
                region = torch.mean(region,dim=(1,2))
                reduced_logits.append(region)
                mask2[xyxy[0]:xyxy[2],xyxy[1]:xyxy[3]]=False

            reduced_logits = torch.stack(reduced_logits)
            fil = ~torch.isnan(reduced_logits[:,0])
            reduced_logits = reduced_logits[fil]
            labels = labels[fil]
            if len(labels)==0:
                print('hmm')
                continue
            logits_neg = torch.mean(logits.permute(1,2,0)[mask2],dim=0,keepdim=True)

            loss_pos += torch.nn.functional.cross_entropy(reduced_logits,labels)
            loss_neg += torch.nn.functional.cross_entropy(logits_neg,neg_label)

            corr_pos += torch.sum(torch.argmax(reduced_logits,dim=1) == labels).item()
            corr_neg += torch.sum(torch.argmax(logits_neg,dim=1) == neg_label).item()
            total_pos += labels.size(0)
            total_neg +=1

    return loss_pos/B,loss_neg/B,corr_pos,total_pos,corr_neg,total_neg

if args.fc:
	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
else:
	optimizer_conv = optim.SGD(model_conv.parameters(), lr=args.lr, momentum=0.9)
	#optimizer_conv = optim.SGD(model_conv.module.linear_aggr.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#print(optimizer_conv.state_dict())
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
if args.resume:
    print('restoring model from checkpoint...')
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model_name))
    model_conv.load_state_dict(checkpoint['model_state_dict'])
    optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

model_conv = model_conv.to(device)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=args.epoch)

