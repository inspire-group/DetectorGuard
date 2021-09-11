
import numpy as np
import argparse
import os
import joblib
parser = argparse.ArgumentParser()

parser.add_argument("--dump_dir",default='pro_dump8',type=str,help='name of dump directory')
parser.add_argument("--imgst_dir",default='mimgst',type=str,help='to get clean evaluation results')
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--w",default=8,type=int,help='window size')
parser.add_argument("--mode",default='clip',choices=['mask','clip'],type=str,help='type of secure PatchGuard aggregation; robust masking or clipping')
parser.add_argument("--m",default=-1,type=int,help='mask size, if mode is mask')
parser.add_argument("--eps",default=3,type=int,help='DBSCAN eps')
parser.add_argument("--ms",default=24,type=int,help='DBSCAN min number of samples')
parser.add_argument("--t",default=32,type=float,help='binarization threshold')
parser.add_argument("--p",default=8,type=int,help='patch size in the feature space')
parser.add_argument("--num_img",default=500,type=int,help='number of images used in this experiment')
parser.add_argument("--loc",default='far',choices=['far','close','over'],type=str,help='far, close, or over patch threat model')

parser.add_argument("--detector",default='yolo',type=str)

args = parser.parse_args()
DATASET = args.dataset
DUMP_DIR = os.path.join(args.dump_dir,DATASET)
SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format(args.detector,args.mode,args.w,args.m,args.t,args.eps,args.ms)
IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)


def get_gt_pro(mode,loc,w,m,p,num,eps,ms,t):
	res_gt = joblib.load(os.path.join(DUMP_DIR,'res_dict_gt_{}_w{}m{}p{}_n{}_{}_{}.z'.format(mode,w,m,p,num,eps,ms)))
	res_gt = res_gt[loc][t]
	num_obj = len(res_gt)
	robust_list = np.array([x[0]>0 for x in res_gt])
	fa_list = np.array([x[3] for x in res_gt])
	tp_list = ~np.array([x[4] for x in res_gt])
	robust_list[fa_list] = False
	num_tp = np.sum(tp_list)
	robust_list[~tp_list]=False

	num_robust = np.sum(robust_list)
	rec_gt = np.mean(joblib.load(os.path.join(IMGST_DIR,'rec.z')))
	print('Evaluation for Perfect Clean Detector')
	print('Clean recall:',rec_gt,'Certified Recall:',num_robust/num_obj)


def get_pro(detector,mode,loc,w,m,p,num,eps,ms,t):
	res_dict = joblib.load(os.path.join(DUMP_DIR,'res_dict_{}_{}_w{}m{}p{}_n{}_{}_{}.z'.format(detector,mode,w,m,p,num,eps,ms)))
	res_list = res_dict[loc][t]
	num_obj = len(res_list)
	base_conf_list=np.linspace(0,0.999,1000)[::-1]
	rec = joblib.load(os.path.join(IMGST_DIR,'rec.z'))
	conf_list_max = np.array([x[0] for x in res_list])

	fa_list = np.array([x[3] for x in res_list])
	tp_list = ~np.array([x[4] for x in res_list])
	conf_list_max[~tp_list]=0
	
	#conf_list_max = conf_list_max[~fa_list]
	conf_list_max[fa_list]=0

	num_robust_list = np.zeros_like(base_conf_list)
	for i,thres in enumerate(base_conf_list):
		num_robust_list[i] = np.sum(conf_list_max>thres)

	rec = np.mean(rec,1) # reduce to averaged recall 
	tmp = np.argsort(rec)

	rec= rec[tmp]
	num_robust_list = num_robust_list[tmp]
	print('Evaluation for {}'.format(detector))
	#print(rec)
	for r in np.arange(0.2,1,0.1): #extremely large or small clean recall might give weird results
		idx = np.searchsorted(rec,r)
		print('Clean recall:',rec[idx],'Certified Recall:',num_robust_list[idx]/num_obj)


if args.detector == 'gt':
	get_gt_pro(args.mode,args.loc,args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t)
else:
	get_pro(args.detector,args.mode,args.loc,args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t)
