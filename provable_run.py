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
parser.add_argument("--dump_dir",default='pro_dump',type=str,help='name of dump directory')
parser.add_argument("--data_dir",default='data',type=str,help='data directory')
parser.add_argument("--det_dir",default='det',type=str,help='directory of base detection outputs')
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--model",default='bagnet33',type=str,help='model name')
parser.add_argument("--w",default=8,type=int,help='window size')
parser.add_argument("--mode",default='clip',choices=['mask','clip'],type=str,help='type of secure PatchGuard aggregation; robust masking or clipping')
parser.add_argument("--m",default=-1,type=int,help='mask size, if mode is mask')
parser.add_argument("--eps",default=3,type=int,help='DBSCAN eps')
parser.add_argument("--ms",default=24,type=int,help='DBSCAN min number of samples')

parser.add_argument("--cache",action='store_true',help='whether to use cached local features (so that no GPU is needed)')
parser.add_argument("--t_min",default=28,type=int,help='smallest binarization threshold in this experiment')
parser.add_argument("--t_max",default=36,type=int,help='largest binarization threshold in this experiment')
parser.add_argument("--onoffset",default=4,type=int,help='a parameter for tuning the boundary between different threat models (see defense.py for more details)')

parser.add_argument("--p",default=8,type=int,help='patch size in the feature space')
parser.add_argument("--num_img",default=100,type=int,help='number of images used in this experiment')

args = parser.parse_args()

DATASET = args.dataset
MODEL_DIR=os.path.join(args.model_dir)
DATA_DIR=os.path.join(args.data_dir,DATASET)
DUMP_DIR = args.dump_dir+str(args.onoffset)
if not os.path.exists(DUMP_DIR):
	os.mkdir(DUMP_DIR)
DUMP_DIR = os.path.join(DUMP_DIR,DATASET)
if not os.path.exists(DUMP_DIR):
	os.mkdir(DUMP_DIR)
MODE = args.mode

MASK_SIZE = args.m
WINDOW_SIZE=args.w
PATCH_SIZE = args.p
PATCH_SHAPE = (PATCH_SIZE,PATCH_SIZE)
NUM_CLS = NUM_CLASS_DICT[DATASET]
CACHE = args.cache

dbscan = DBSCAN(eps=args.eps, min_samples=args.ms)

# set up dataset
if DATASET == 'voc':
	transforms = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	val_dataset = VOCDetection(DATA_DIR,image_set='test',year='2007',transforms=transforms)
elif DATASET == 'coco':
	val_dataset = get_coco(DATA_DIR,image_set='val')
elif DATASET == 'kitti':
	val_dataset = Kitti2DDetection(DATA_DIR,image_set='val')
np.random.seed(233333333)#233333
idxs=np.arange(len(val_dataset))
np.random.shuffle(idxs)
idxs=idxs[:args.num_img]

if CACHE: #the provable analysis does not require gpus if we use the cached feature map 
	feature_list = joblib.load(os.path.join(DATA_DIR,'feature_list_{}_{}.z'.format(args.model,DATASET)))
	target_list = joblib.load(os.path.join(DATA_DIR,'target_list_{}_{}.z'.format(args.model,DATASET)))
	feature_list = [feature_list[i] for i in idxs]
	target_list = [target_list[i] for i in idxs]
	val_loader = zip(feature_list,target_list)
	if 'bagnet33' in args.model:
		FM_SIZE = FM_SIZE_DICT[DATASET] #hard-coded feature map size for now
		RF_SIZE = 33
else:
	val_dataset = torch.utils.data.Subset(val_dataset, idxs)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,shuffle=False,pin_memory=True,collate_fn=collate_fn,num_workers=2)
	device='cuda'
	if 'bagnet33' in args.model:
		bagnet_model = utils.bagnet.bagnet33(aggregation='none')
		FM_SIZE = FM_SIZE_DICT[DATASET] #hard-coded feature map size for now
		RF_SIZE = 33

	num_ftrs = bagnet_model.fc.in_features
	bagnet_model.fc = torch.nn.Linear(num_ftrs, NUM_CLS+1)
	bagnet_model = torch.nn.DataParallel(bagnet_model)
	checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'{}_.pth'.format(DATASET)))
	bagnet_model.load_state_dict(checkpoint['model_state_dict'])
	bagnet_model = bagnet_model.to(device)
	bagnet_model.eval()

det_raw_yolo = read_det(os.path.join('det',DATASET,'det_img_yolo'))
det_raw_frcnn = read_det(os.path.join('det',DATASET,'det_img_frcnn'))

t_list = np.arange(args.t_min,args.t_max,1) if DATASET!='kitti' else np.arange(args.t_min,args.t_max,0.5)

res_dict_gt = {'over':{thres:[] for thres in t_list},'close':{thres:[] for thres in t_list},'far':{thres:[] for thres in t_list}}
res_dict_yolo = {'over':{thres:[] for thres in t_list},'close':{thres:[] for thres in t_list},'far':{thres:[] for thres in t_list}}
res_dict_frcnn = {'over':{thres:[] for thres in t_list},'close':{thres:[] for thres in t_list},'far':{thres:[] for thres in t_list}}

locs = ['over','close','far']

cnt=0
for img,targets in tqdm(val_loader):

	assert len(targets[0]['labels'])>0

	img_id = targets[0]['img_id']

	labels = targets[0]['labels'] = targets[0]['labels'].numpy()
	
	if not CACHE:
		local_features = bagnet_model(img.cuda()).detach().cpu().numpy().squeeze().transpose(1,2,0)		
	else:
		local_features = img

	# first get clean prediction
	local_features = np.clip(local_features,0,np.inf)
	if MODE == 'mask':
		raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],mode=MODE,mask_size=MASK_SIZE)
	elif MODE == 'clip':
		raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],mode=MODE)
	
	# get pixel-space and feature-space bounding boxes (for Perfect Clean Detector)
	img_bboxes_gt = targets[0]['bbox']
	fm_bboxes_gt = bboxes_img2fm(img_bboxes_gt,FM_SIZE,RF_SIZE)

	# get pixel-space and feature-space bounding boxes (for YOLO and FRCNN)
	img_bboxes_yolo,confs_yolo,labels_yolo = rescale_det(det_raw_yolo[img_id],targets[0]['ratio'],targets[0]['pad'])
	img_bboxes_frcnn,confs_frcnn,labels_frcnn = rescale_det(det_raw_frcnn[img_id],targets[0]['ratio'],targets[0]['pad'])

	fm_bboxes_yolo = bboxes_img2fm(img_bboxes_yolo,FM_SIZE,RF_SIZE)
	fm_bboxes_frcnn = bboxes_img2fm(img_bboxes_frcnn,FM_SIZE,RF_SIZE)

	alert_dict_gt = {}
	alert_dict_yolo = {}
	alert_dict_frcnn = {}
	fn_flgs_dict = {}
	# get clean information: whether defense triggers a false alert on clean image
	for j,thres in enumerate(t_list):
		obj_map = (raw_obj_map > WINDOW_SIZE**2 *thres).astype(float)
		fn_flgs,alert_flg = explainer(fm_bboxes_gt,obj_map,dbscan)
		alert_dict_gt[thres]=alert_flg
		fn_flgs_dict[thres]=fn_flgs
		_,alert_flg = explainer(fm_bboxes_yolo,obj_map,dbscan)
		alert_dict_yolo[thres]=alert_flg

		_,alert_flg = explainer(fm_bboxes_frcnn,obj_map,dbscan)
		alert_dict_frcnn[thres]=alert_flg


	raw_obj_map_cache={}
	# check robustness
	for i in range(len(fm_bboxes_gt)):
		vul_dict_gt={'over':{thres:False for thres in t_list},'close':{thres:False for thres in t_list},'far':{thres:False for thres in t_list}}

		box_gt = fm_bboxes_gt[i]
		# get the max confidence threshold for base detector to detect this object; if not detected, get -2
		# to determine if the object can be detected in the clean setting
		maxconf_yolo = get_box(img_bboxes_yolo,img_bboxes_gt[i],confs_yolo) 
		maxconf_frcnn = get_box(img_bboxes_frcnn,img_bboxes_gt[i],confs_frcnn)

		# get location list for different threat models
		loc_dict = gen_patch_loc(box_gt,PATCH_SHAPE,FM_SIZE,pskip=1,onoffset=args.onoffset)
		for loc in ['over','close','far']:
			loc_list = loc_dict[loc]
			for (patch_x,patch_y) in loc_list:
				patch_box = np.array([patch_x,patch_y,patch_x+PATCH_SHAPE[0],patch_y+PATCH_SHAPE[1]])
				if (patch_x,patch_y) not in raw_obj_map_cache:
					if MODE == 'mask':
						raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],mode=MODE,patch_box_abs=[patch_box],mask_size=MASK_SIZE)
					elif MODE == 'clip':
						raw_obj_map = gen_obj_map(local_features,window_size=WINDOW_SIZE,pad=targets[0]['pad'],patch_box_abs=[patch_box],mode=MODE)
				else:
					raw_obj_map = raw_obj_map_cache[(patch_x,patch_y)]

				for thres in t_list:
					if check_vul(raw_obj_map,thres,box_gt,dbscan,WINDOW_SIZE):
						vul_dict_gt[loc][thres]=True
				if np.all(list(vul_dict_gt[loc].values())):# vulerable for all binarization thresholds, certification fails for all thresholds
					break

			p_flag = len(loc_list)>0
			for thres in t_list:
				tmp = -1 if vul_dict_gt[loc][thres] else 1
				res = [tmp,img_id,img_bboxes_gt[i],alert_dict_gt[thres],fn_flgs_dict[thres][i],labels[i],p_flag]
				res_dict_gt[loc][thres].append(res)

				tmp = -1 if vul_dict_gt[loc][thres] else maxconf_yolo
				res = [tmp,img_id,img_bboxes_gt[i],alert_dict_yolo[thres],fn_flgs_dict[thres][i],labels[i],p_flag]
				res_dict_yolo[loc][thres].append(res)
				
				tmp = -1 if vul_dict_gt[loc][thres] else maxconf_frcnn
				res = [tmp,img_id,img_bboxes_gt[i],alert_dict_frcnn[thres],fn_flgs_dict[thres][i],labels[i],p_flag]
				res_dict_frcnn[loc][thres].append(res)

joblib.dump(res_dict_yolo,os.path.join(DUMP_DIR,'res_dict_yolo_{}_w{}m{}p{}_n{}_{}_{}.z'.format(args.mode,args.w,args.m,PATCH_SIZE,args.num_img,args.eps,args.ms)))
joblib.dump(res_dict_frcnn,os.path.join(DUMP_DIR,'res_dict_frcnn_{}_w{}m{}p{}_n{}_{}_{}.z'.format(args.mode,args.w,args.m,PATCH_SIZE,args.num_img,args.eps,args.ms)))
joblib.dump(res_dict_gt,os.path.join(DUMP_DIR,'res_dict_gt_{}_w{}m{}p{}_n{}_{}_{}.z'.format(args.mode,args.w,args.m,PATCH_SIZE,args.num_img,args.eps,args.ms)))

