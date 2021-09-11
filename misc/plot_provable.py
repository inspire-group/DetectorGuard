
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import joblib
parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='pro_dump',type=str,help='name of dump directory')
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


args = parser.parse_args()
DATASET = args.dataset
DUMP_DIR = os.path.join(args.dump_dir,DATASET)


font = {'family' : 'serif',
		#'weight' : 'bold',
		'size'   : 12}

matplotlib.rc('font', **font)



lw = 2

if DATASET == 'voc':
	NUM_CLS = 20
	IMG_AREA = 416*416
	CLS_NAME = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
	yolo_cut = 0.933
	frcnn_cut = 0.997
elif DATASET == 'coco':
	NUM_CLS = 80
	IMG_AREA = 416*416
	yolo_cut = 0.9
	frcnn_cut = 0.997
elif DATASET == 'kitti':
	NUM_CLS = 80
	IMG_AREA = 740*224
	yolo_cut = 0.0
	frcnn_cut = 0.0
	
def get_gt_pro_plot(mode,loc,w,m,p,num,eps,ms,t):
	res_gt = joblib.load(os.path.join(DUMP_DIR,'res_dict_gt_{}_w{}m{}p{}_n{}_{}_{}.z'.format(mode,w,m,p,num,eps,ms)))
	res_gt = res_gt[loc][t]
	num_obj = len(res_gt)
	robust_list = np.array([x[0]>0 for x in res_gt])
	fa_list = np.array([x[3] for x in res_gt])
	tp_list = ~np.array([x[4] for x in res_gt])
	cls_list = np.array([x[5] for x in res_gt])

	robust_list[fa_list] = False
	num_tp = np.sum(tp_list)
	robust_list[~tp_list]=False
	num_robust = np.sum(robust_list)

	obj_size = np.array([(x[2][2]-x[2][0]) *(x[2][3]-x[2][1]) for x in res_gt])
	robust_cls = np.array([np.mean(robust_list[cls_list==x]) for x in range(NUM_CLS)])
	size_cls = np.array([np.mean(obj_size[cls_list==x]) for x in range(NUM_CLS)])
	size_cls = size_cls/(IMG_AREA)
	return num_robust/num_obj *100,robust_cls*100,size_cls*100

def plt_size(robust_cls,size_cls,name):
	if 'far' in name:
		loc = 'far'
	elif 'close' in name:
		loc = 'close'
	elif 'over' in name:
		loc = 'over'
	fig, ax1 = plt.subplots()
	br1 = np.arange(NUM_CLS)*1.1
	width = 0.45
	colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
	ax1.bar(br1 - width/2, robust_cls, width, label='PCD-{}'.format(loc),color = colors[0])
	ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
	ax2.bar(br1 + width/2, size_cls, width, label='Average object size',color = colors[1])
	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2,loc='upper left')

	ax1.set_ylabel('CR (%)')    
	ax2.set_ylabel('Object size (%)')  # we already handled the x-label with ax1
	ax1.set_xticks(br1)
	ax1.set_xticklabels(CLS_NAME,rotation=45,ha='right', rotation_mode='anchor',fontsize=10)
	ax1.set_yticks(np.arange(0,71,10))
	ax1.set_xlim([-1,22])
	plt.tight_layout()

	plt.savefig('{}.png'.format(name))
	plt.savefig('{}.pdf'.format(name))
	plt.close()

###############################################GT########################################################
if DATASET == 'voc':
	cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='far',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
	plt_size(robust_cls,size_cls,'per_class_far')
	cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='close',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
	plt_size(robust_cls,size_cls,'per_class_close')
	cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='over',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
	plt_size(robust_cls,size_cls,'per_class_over')
SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format('gt',args.mode,args.w,args.m,args.t,args.eps,args.ms)
IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)
REC_GT = np.mean(joblib.load(os.path.join(IMGST_DIR,'rec.z'))) *100

cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='far',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
plt.plot([REC_GT],[cr],'o',label='PCD-far',markersize=6)
#print('GT', 'far', cr)

cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='close',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
plt.plot([REC_GT],[cr],'o',label='PCD-close',markersize=6)
#print('GT', 'close', cr)

cr,robust_cls,size_cls = get_gt_pro_plot(mode=args.mode,loc='over',w=args.w,m=args.m,p=args.p,num=args.num_img,eps=args.eps,ms=args.ms,t=args.t)
plt.plot([REC_GT],[cr],'o',label='PCD-over',markersize=6)
#print('GT', 'in', cr)

#################################YOLO#################################################
def get_pro_plot(detector,mode,loc,w,m,p,num,eps,ms,t,conf_cut):
	SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format(detector,mode,w,m,t,eps,ms)
	IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)

	res_dict = joblib.load(os.path.join(DUMP_DIR,'res_dict_{}_{}_w{}m{}p{}_n{}_{}_{}.z'.format(detector,mode,w,m,p,num,eps,ms)))
	res_list = res_dict[loc][t]
	num_obj = len(res_list)
	base_conf_list=np.linspace(0,0.999,1000)[::-1]
	rec = joblib.load(os.path.join(IMGST_DIR,'rec.z'))
	conf_list_max = np.array([x[0] for x in res_list])
	#print(np.mean(joblib.load(os.path.join(IMGST_DIR,'rec.z')),1))
	#print(np.mean(joblib.load(os.path.join(IMGST_DIR,'prec.z')),1))

	fa_list = np.array([x[3] for x in res_list])
	tp_list = ~np.array([x[4] for x in res_list])
	conf_list_max[~tp_list]=0
	
	conf_list_max = conf_list_max[~fa_list]

	num_robust_list = np.zeros_like(base_conf_list)
	for i,thres in enumerate(base_conf_list):
		num_robust_list[i] = np.sum(conf_list_max>thres)
	#tmp = base_conf_list < conf_cut # remove points for high confidence threshold (a confidence threshold that is too high can result in no predicted bounding box)
	#num_robust_list = num_robust_list[tmp]
	#rec = rec[tmp]
	#rec = np.nanmean(np.where(rec!=0,rec,np.nan),1) # reduce to averaged recall 

	rec = np.mean(rec,1) # reduce to averaged recall 

	tmp = np.argsort(rec)
	rec= rec[tmp]
	num_robust_list = num_robust_list[tmp]
	
	#print(SETTING)
	#for r in np.arange(0.2,1,0.1): #extremely large or small clean recall might give weird results
	#	idx = np.searchsorted(rec,r)
	#	print('Clean recall:',rec[idx],'Certified Recall:',num_robust_list[idx]/num_obj)

	tmp = rec > 0.11
	rec = rec[tmp]
	num_robust_list = num_robust_list[tmp]
	return num_robust_list/num_obj*100,rec*100



ss = 5

rc_list,rec = get_pro_plot('yolo',args.mode,'far',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,yolo_cut)
plt.plot(rec[::ss],rc_list[::ss],label='YOLO-far',linewidth=lw,linestyle='-')

rc_list,rec = get_pro_plot('yolo',args.mode,'close',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,yolo_cut)
plt.plot(rec[::ss],rc_list[::ss],label='YOLO-close',linewidth=lw,linestyle='-')

rc_list,rec = get_pro_plot('yolo',args.mode,'over',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,yolo_cut)
plt.plot(rec[::ss],rc_list[::ss],label='YOLO-over',linewidth=lw,linestyle='-')

#---------------------------------------------------------------------------------------------------

rc_list,rec = get_pro_plot('frcnn',args.mode,'far',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,frcnn_cut)
plt.plot(rec[::ss],rc_list[::ss],label='FRCNN-far',linewidth=lw,linestyle='-.')

rc_list,rec = get_pro_plot('frcnn',args.mode,'close',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,frcnn_cut)
plt.plot(rec[::ss],rc_list[::ss],label='FRCNN-close',linewidth=lw,linestyle='-.')

rc_list,rec = get_pro_plot('frcnn',args.mode,'over',args.w,args.m,args.p,args.num_img,args.eps,args.ms,args.t,frcnn_cut)
plt.plot(rec[::ss],rc_list[::ss],label='FRCNN-over',linewidth=lw,linestyle='-.')


###############################################################################################
plt.legend(loc='lower left')

plt.xlabel('Clean Recall (%)')
plt.ylabel('Certified Recall (%)')

plt.grid()

plt.xticks(np.arange(10,105,10))
plt.xlim([6,104])
if DATASET == 'voc':
	plt.yticks(np.arange(0.0,31,5))
	plt.ylim([-1,31])

elif DATASET == 'coco':
	plt.yticks(np.arange(0.0,16,2.5))
	plt.ylim([-1,16])


plt.tight_layout()

plt.savefig('pro_{}_{}.pdf'.format(args.dataset,args.mode))
plt.savefig('pro_{}_{}.png'.format(args.dataset,args.mode))
plt.close()

