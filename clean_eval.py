from utils.eval_utils import *
from tqdm import tqdm
import numpy as np
import argparse
import os
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",default='data',type=str,help='data directory (to get ground truth for evaluation)')
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--det_dir",default='det',type=str)
parser.add_argument("--imgst_dir",default='mimgst',type=str)
parser.add_argument("--w",default=8,type=int,help='window size')
parser.add_argument("--mode",default='clip',choices=['mask','clip'],type=str,help='type of secure PatchGuard aggregation; robust masking or clipping')
parser.add_argument("--m",default=-1,type=int,help='mask size, if mode is mask')
parser.add_argument("--t",default=32,type=float,help='binarization threshold')
parser.add_argument("--eps",default=3,type=int,help='DBSCAN eps')
parser.add_argument("--ms",default=24,type=int,help='DBSCAN min number of samples')
parser.add_argument("--detector",default='yolo',choices=['yolo','gt','frcnn'],type=str,help='type of base detector')
parser.add_argument("--vanilla",action='store_true',help='evaluate undefended Base Detector for AP comparison')
parser.add_argument("--dump",action='store_true',help='dump the evaluation results (for further analysis)')

args = parser.parse_args()
DATASET = args.dataset
VANILLA = args.vanilla 
GT = args.detector == 'gt'
DUMP = args.dump
DET_PATH = os.path.join(args.det_dir,DATASET,'det_cls_{}'.format(args.detector),'{}.txt')
if DATASET == 'voc':
	DATA_ROOT = os.path.join(args.data_dir,'voc','VOCdevkit','VOC2007')
	ANNO_PATH = os.path.join(DATA_ROOT,'Annotations/{}.xml')
	classes_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
elif DATASET == 'coco':
	ANNO_PATH = os.path.join(args.data_dir,'coco','annotations','instances_val2017.json') 
	coco = COCO(ANNO_PATH)
	ANNO_PATH = coco
	with open(os.path.join(args.data_dir,'coco','coco.txt')) as f:
		lines= f.readlines()
	classes_name = [x.strip() for x in lines]
elif DATASET == 'kitti':
	DATA_ROOT = os.path.join(args.data_dir,'kitti')
	ANNO_PATH = os.path.join(DATA_ROOT,'label_2','{}.txt')
	classes_name = ['car','person','cyclist']

if VANILLA:
	IMGST_PATH = os.path.join(DATA_ROOT,'ImageSets','Main','test.txt') if DATASET == 'voc' else None
	if DATASET == 'kitti':
		IMGST_PATH = os.path.join(DATA_ROOT,'val.txt')
	FAFILE_PATH = None
else:	
	SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format(args.detector,args.mode,args.w,args.m,args.t,args.eps,args.ms)
	IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)
	if GT:
		IMGST_PATH = os.path.join(IMGST_DIR,'imgst.txt') 
		FAFILE_PATH = os.path.join(IMGST_DIR,'imgfa.txt')	
	else:
		IMGST_PATH = os.path.join(IMGST_DIR,'imgst{:.3f}.txt') 
		FAFILE_PATH = os.path.join(IMGST_DIR,'imgfa{:.3f}.txt')

if GT:
	base_conf_list=[0]
else:
	base_conf_list=np.linspace(0,0.999,1000)[::-1]
prec = np.zeros([len(base_conf_list),len(classes_name)]) # precision at different confidence threshold of Base Detector
rec = np.zeros([len(base_conf_list),len(classes_name)]) # recall at different confidence threshold of Base Detector
fa_list = np.zeros([len(base_conf_list)]) # False alert rate at different confidence threshold of Base Detector

for i,base_conf in tqdm(enumerate(base_conf_list)):
	imagesetfile = IMGST_PATH if VANILLA else IMGST_PATH.format(base_conf) # the file containing a list of image ids that DetectorGuard does not issue an alert
	fafile = None if VANILLA else FAFILE_PATH.format(base_conf) # the file containing a list of image ids that DetectorGuard issues an alert
	for j,clss in enumerate(classes_name): # get precision and recall for each class
		r,p,fa = eval_prec_rec(dataset=args.dataset,detpath=DET_PATH,annopath=ANNO_PATH,
		imagesetfile=imagesetfile,fafile=fafile,
		classname=clss,conf_thres=base_conf)
		prec[i,j]=p
		rec[i,j]=r
		fa_list[i]=fa

if GT:
	print('Results for Perfect Clean Detector')
	mAP = np.mean(rec[0]) # the precision is also 1 for GT
	fa = fa_list[0]
	print('mAP:',mAP)
	print('FAR:',fa)
	if DUMP:
		joblib.dump(prec[0,:],os.path.join(IMGST_DIR,'prec.z'))
		joblib.dump(rec[0,:],os.path.join(IMGST_DIR,'rec.z'))
		joblib.dump(fa_list[0],os.path.join(IMGST_DIR,'fa_list.z'))
		joblib.dump(mAP,os.path.join(IMGST_DIR,'mAP.z'))
else:
	print('Results for {}'.format(args.detector))
	print('Per-class AP:')
	mAP=[]
	for j in range(len(classes_name)):
		ap = voc_ap(rec[:,j].reshape(-1),prec[:,j].reshape(-1)) # per-class AP
		print(j,classes_name[j],ap)
		mAP.append(ap)
	print('mAP:',np.mean(mAP))

	if DUMP:
		if VANILLA:
			joblib.dump(prec,os.path.join(args.imgst_dir,args.dataset,'prec_vanilla_{}.z'.format(args.detector)))
			joblib.dump(rec,os.path.join(args.imgst_dir,args.dataset,'rec_vanilla_{}.z'.format(args.detector)))
			joblib.dump(fa_list,os.path.join(args.imgst_dir,args.dataset,'fa_list_vanilla_{}.z'.format(args.detector)))
		else:
			joblib.dump(prec,os.path.join(IMGST_DIR,'prec.z'))
			joblib.dump(rec,os.path.join(IMGST_DIR,'rec.z'))
			joblib.dump(fa_list,os.path.join(IMGST_DIR,'fa_list.z'))
	

	rec = np.mean(rec,1) # reduce to averaged recall 
	tmp = np.argsort(rec)
	rec= rec[tmp]
	fa_list = fa_list[tmp]
	for r in np.arange(0.2,1,0.1): 
		idx = np.searchsorted(rec,r)
		print('Clean recall:',rec[idx],'FAR:',fa_list[idx])





