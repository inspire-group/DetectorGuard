import torch
import torchvision.transforms as T

import os
import time
from tqdm import tqdm
import joblib
from sklearn.cluster import DBSCAN

import argparse
import numpy as np

import utils.bagnet
from utils.defense import *
from utils.box_utils import *
from utils.dataset import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help='model directory')
parser.add_argument("--data_dir",default='data',type=str,help='data directory')
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--det_dir",default='det',type=str,help='directory of base detection outputs')
parser.add_argument("--output_dir",default='mimgst',type=str,help='output directory')
parser.add_argument("--model",default='bagnet33',type=str,help='model name')
parser.add_argument("--w",default=8,type=int,help='window size')
parser.add_argument("--mode",default='clip',choices=['mask','clip'],type=str,help='type of secure PatchGuard aggregation; robust masking or clipping')
parser.add_argument("--m",default=-1,type=int,help='mask size, if mode is mask')
parser.add_argument("--t",default=32,type=float,help='binarization threshold')
parser.add_argument("--eps",default=3,type=int,help='DBSCAN eps')
parser.add_argument("--ms",default=24,type=int,help='DBSCAN min number of samples')
parser.add_argument("--detector",default='yolo',choices=['yolo','gt','frcnn'],type=str,help='type of base detector')

args = parser.parse_args()
DATASET = args.dataset
GT = args.detector == 'gt'
MODEL_DIR = args.model_dir
DATA_DIR = os.path.join(args.data_dir,DATASET)
DET_DIR = os.path.join(args.det_dir,DATASET)

if not os.path.exists(args.output_dir):
	os.mkdir(args.output_dir)
OUT_DIR = os.path.join(args.output_dir,DATASET)
if not os.path.exists(OUT_DIR):
	os.mkdir(OUT_DIR)

MASK_SIZE = args.m
WINDOW_SIZE=args.w
NUM_CLS = NUM_CLASS_DICT[DATASET]
MODE = args.mode

dbscan = DBSCAN(eps=args.eps, min_samples=args.ms)

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


if GT: # use ground truth bounding boxes (i.e., Perfect Clean Detector in the paper)
	base_conf_list=[0] # no confidence thresthold needed for Base Detector
else: # use vanilla Base Detector (e.f., YOLO, Faster-RCNN)
	base_conf_list=np.linspace(0,0.999,1000) # list of Base Detector output confidence thresholds
	# read per-image detection results of Base Detector	
	input_dir = os.path.join(DET_DIR,'det_img_{}'.format(args.detector))
	det_raw = read_det(input_dir)

OUT_DIR = os.path.join(OUT_DIR,'{}_{}_w{}m{}t{:.1f}_{}_{}'.format(args.detector,args.mode,WINDOW_SIZE,MASK_SIZE,args.t,args.eps,args.ms))
os.mkdir(OUT_DIR)

alert_dict = {base_conf:0 for base_conf in base_conf_list} # log how many clean images for which DetectorGuard issues a (false) alert (not used for the paper evaluation)
corr_dict = {base_conf:0 for base_conf in base_conf_list} # log how many clean images for which DetectorGuard correctly does not issue an alert (not used for the paper evaluation)


for img,targets in tqdm(val_loader):
	assert len(targets[0]['labels']) != 0 # should not fail this assertion if data is correctly loaded

	img_id = targets[0]['img_id']

	### Objectness Predictor
	# obtain feature map
	local_features = bagnet_model(img.cuda()).detach().cpu().numpy().squeeze().transpose(1,2,0)
	local_features = np.clip(local_features,0,np.inf)
	# use feature-space slide window for robust window classification
	if MODE == 'mask':
		raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],mode=MODE,mask_size=MASK_SIZE)
	elif MODE == 'clip':
		raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],mode=MODE)
	# binarization
	obj_map = (raw_obj_map > WINDOW_SIZE**2 *args.t).astype(float) 

	### Objectness Explainer
	if GT:
		img_bboxes = targets[0]['bbox']
	else:
		# rescale base detector output; to be consistent with BagNet images
		# note that the output of Base Detector is for the original image resolution, while in BagNet we use 416x416 images
		det = det_raw[img_id]
		img_bboxes,confs,labels = rescale_det(det,targets[0]['ratio'],targets[0]['pad'])

	# map pixel coordinates to feature coordinates
	fm_bboxes = bboxes_img2fm(img_bboxes,FM_SIZE,RF_SIZE)
	if GT:
		# perform objectness explaining (i.e., Objectness Explainer)
		_,alert_flg = explainer(fm_bboxes,obj_map,dbscan)
		# log the image_id to the alert file or the correct file
		if alert_flg:
			alert_dict[0]+=1
			with open(os.path.join(OUT_DIR,'imgfa.txt'),'a') as f:
				f.write('{}\n'.format(img_id))	
		else:
			corr_dict[0]+=1
			with open(os.path.join(OUT_DIR,'imgst.txt'),'a') as f:
				f.write('{}\n'.format(img_id))
	else:
		for base_conf in base_conf_list:
			# perform objectness explaining (i.e., Objectness Explainer)
			tmp = confs > base_conf
			_,alert_flg = explainer(fm_bboxes[tmp],obj_map,dbscan)
			# log the image_id to the alert file or the correct file
			if alert_flg:
				with open(os.path.join(OUT_DIR,'imgfa{:.3f}.txt'.format(base_conf)),'a') as f:
					f.write('{}\n'.format(img_id))		
				alert_dict[base_conf]+=1
			else:
				with open(os.path.join(OUT_DIR,'imgst{:.3f}.txt'.format(base_conf)),'a') as f:
					f.write('{}\n'.format(img_id))
				corr_dict[base_conf]+=1

print('Statistics for false alert rates')
for base_conf in base_conf_list:
	total = corr_dict[base_conf]+alert_dict[base_conf]
	print('Confidence:{:.3f},FAR:{:.3f}'.format(base_conf,alert_dict[base_conf]/total))


