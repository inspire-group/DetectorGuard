import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import joblib


parser = argparse.ArgumentParser()
parser.add_argument("--imgst_dir",default='mimgst',type=str)
parser.add_argument("--dataset",default='voc',choices=['voc','coco','kitti'],type=str,help='dataset name')
parser.add_argument("--mode",default='clip',choices=['mask','clip'],type=str,help='type of secure PatchGuard aggregation; robust masking or clipping')
parser.add_argument("--w",default=8,type=int)
parser.add_argument("--m",default=-1,type=int)
parser.add_argument("--t",default=32,type=float)
parser.add_argument("--eps",default=3,type=int,help='DBSCAN eps')
parser.add_argument("--ms",default=24,type=int,help='DBSCAN min number of samples')

args = parser.parse_args()


DATASET = args.dataset
font = {'family' : 'serif',
        #'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


########################################GT################################
SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format('gt',args.mode,args.w,args.m,args.t,args.eps,args.ms)
IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)

prec = np.mean(joblib.load(os.path.join(IMGST_DIR,'prec.z')))
rec = np.mean(joblib.load(os.path.join(IMGST_DIR,'rec.z')))
gt_fa_list = joblib.load(os.path.join(IMGST_DIR,'fa_list.z'))

plt.plot([100],[100],'o',label='Precision-PCD-vanilla',markersize=6)

plt.plot([rec*100],[prec*100],'x',label='Precision-PCD-defended',markersize=6)
gt_rec = rec
###############################################################################



def process(prec,rec,fa_list,conf_cut):
	thres_list=np.linspace(0,0.999,1000)[::-1]
	tmp = thres_list <  conf_cut
	thres_list = thres_list[tmp]
	prec = prec[tmp]
	rec = rec[tmp]
	#prec = np.nanmean(np.where(prec!=0,prec,np.nan),1)
	#rec = np.nanmean(np.where(rec!=0,rec,np.nan),1)
	prec = np.mean(prec,1)
	rec = np.mean(rec,1)
	fa_list = fa_list[tmp]
	tmp = np.argsort(rec)
	rec= rec[tmp]
	prec= prec[tmp]
	fa_list= fa_list[tmp]
	return prec,rec,fa_list
lw = 2

if DATASET == 'voc':
	yolo_cut = 0.945
	frcnn_cut = 0.999
elif DATASET =='coco':
	yolo_cut = 0.9
	frcnn_cut = 0.997
elif DATASET == 'kitti':
	yolo_cut = 0.925
	frcnn_cut =  0.9999
######################YOLO########################################################
prec = joblib.load(os.path.join(args.imgst_dir,args.dataset,'prec_vanilla_{}.z'.format('yolo')))
rec = joblib.load(os.path.join(args.imgst_dir,args.dataset,'rec_vanilla_{}.z'.format('yolo')))
fa_list = joblib.load(os.path.join(args.imgst_dir,args.dataset,'fa_list_vanilla_{}.z'.format('yolo')))

prec,rec,fa_list = process(prec,rec,fa_list,yolo_cut)
plt.plot(rec*100,prec*100,label='Precision-YOLO-vanilla',linestyle='-',linewidth=lw)


#####################YOLO_DPG########################################################
SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format('yolo',args.mode,args.w,args.m,args.t,args.eps,args.ms)
IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)

prec = joblib.load(os.path.join(IMGST_DIR,'prec.z'))
rec = joblib.load(os.path.join(IMGST_DIR,'rec.z'))
fa_list = joblib.load(os.path.join(IMGST_DIR,'fa_list.z'))

prec,rec,fa_list = process(prec,rec,fa_list,yolo_cut)
plt.plot(rec*100,prec*100,label='Precision-YOLO-defended',linestyle='-.',linewidth=lw)

yolo_fa_list= fa_list.copy()
yolo_rec = rec.copy()

######################FRCNN########################################################
prec = joblib.load(os.path.join(args.imgst_dir,args.dataset,'prec_vanilla_{}.z'.format('frcnn')))
rec = joblib.load(os.path.join(args.imgst_dir,args.dataset,'rec_vanilla_{}.z'.format('frcnn')))
fa_list = joblib.load(os.path.join(args.imgst_dir,args.dataset,'fa_list_vanilla_{}.z'.format('frcnn')))

prec,rec,fa_list = process(prec,rec,fa_list,frcnn_cut)
plt.plot(rec*100,prec*100,label='Precision-FRCNN-vanilla',linestyle='--',linewidth=lw)


############FRCNN_DPG###############################################

SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format('frcnn',args.mode,args.w,args.m,args.t,args.eps,args.ms)
IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)

prec = joblib.load(os.path.join(IMGST_DIR,'prec.z'))
rec = joblib.load(os.path.join(IMGST_DIR,'rec.z'))
fa_list = joblib.load(os.path.join(IMGST_DIR,'fa_list.z'))

prec,rec,fa_list = process(prec,rec,fa_list,frcnn_cut)
plt.plot(rec*100,prec*100,label='Precision-FRCNN-defended',linestyle='-.',linewidth=lw)

frcnn_fa_list= fa_list.copy()
frcnn_rec = rec.copy()


############FAR##########################################################

plt.plot([gt_rec*100],[gt_fa_list*100],'D',label='FAR-PCD-defended',markersize=6)

plt.plot(yolo_rec*100,yolo_fa_list*100,label='FAR-YOLO-defended',linestyle='-',linewidth=lw)
plt.plot(frcnn_rec*100,frcnn_fa_list*100,label='FAR-FRCNN-defended',linestyle='-.',linewidth=lw)


plt.legend(loc='center left')
plt.grid()
plt.xlabel('Recall (%)')
plt.ylabel('Precision / FAR (%)')

plt.xticks(np.arange(10,110,10))
plt.yticks(np.arange(0,110,10))

#plt.xlim([0,1])
#plt.ylim([0,1])
plt.tight_layout()
plt.savefig('clean_{}_{}.png'.format(DATASET,args.mode))
plt.savefig('clean_{}_{}.pdf'.format(DATASET,args.mode))
plt.close()


