from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import joblib 
parser = argparse.ArgumentParser()
parser.add_argument("--imgst_dir",default='mimgst',type=str)
parser.add_argument("--w",default=8,type=int)
parser.add_argument("--m",default=-1,type=int)
parser.add_argument("--p",default=8,type=int)
parser.add_argument("--t",default=32,type=float)
parser.add_argument("--para",default='w',type=str)
parser.add_argument("--num_img",default=500,type=int)
parser.add_argument("--dump_dir",default='pro_dump8',type=str,help='name of dump directory')
parser.add_argument("--bold",action='store_true',help='bold')

args = parser.parse_args()

DUMP_DIR = args.dump_dir
DATASET = 'voc'

BOLD=args.bold

if BOLD:
    font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 14}
else:
    font = {'family' : 'serif',
        #'weight' : 'bold',
        'size'   : 14}  
matplotlib.rc('font', **font)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

###############################################GT########################################################

def get_gt_pro(loc,w,m,t,p,num,eps=3,ms=24):
    res_gt = joblib.load(os.path.join(DUMP_DIR,DATASET,'res_dict_gt_{}_w{}m{}p{}_n{}_{}_{}.z'.format('clip',w,m,p,num,eps,ms)))
    res_gt = res_gt[loc][t]
    num_obj = len(res_gt)
    robust_list = np.array([x[0]>0 for x in res_gt])
    fa_list = np.array([x[3] for x in res_gt])
    tp_list = ~np.array([x[4] for x in res_gt])
    robust_list[fa_list] = False
    num_tp = np.sum(tp_list)
    robust_list[~tp_list]=False
    num_robust = np.sum(robust_list)
    return num_robust/num_obj *100

def get_gt_clean(w,m,t,e=3,ms=24):
    SETTING = '{}_{}_w{}m{}t{:.1f}_{}_{}'.format('gt','clip',w,m,t,e,ms)
    IMGST_DIR = os.path.join(args.imgst_dir,DATASET,SETTING)

    prec = joblib.load(os.path.join(IMGST_DIR,'prec.z'))
    rec = joblib.load(os.path.join(IMGST_DIR,'rec.z'))
    fa_list = joblib.load(os.path.join(IMGST_DIR,'fa_list.z'))
    mAP = joblib.load(os.path.join(IMGST_DIR,'mAP.z'))
    
    return np.mean(prec),np.mean(rec),fa_list,np.mean(mAP)


###########################################WINDOW SIZE##########################################
ax1_y = np.arange(0,45,5)
ax2_y = np.arange(94,101,1)

if args.para == 'w':
    cr_in = []
    cr_close=[]
    cr_far=[]
    max_cr = []
    w_list = [4,5,6,7,8,9,10,11,12]
    mAP = []
    prec = []
    rec = []
    fa = []
    for w in w_list:
        cr = get_gt_pro('over',w=w,m=-1,p=8,num=args.num_img,t=args.t)
        cr_in.append(cr)
        cr = get_gt_pro('close',w=w,m=-1,p=8,num=args.num_img,t=args.t)
        cr_close.append(cr)
        cr = get_gt_pro('far',w=w,m=-1,p=8,num=args.num_img,t=args.t)
        cr_far.append(cr)
        
        p,r,f,m=get_gt_clean(w=w,m=-1,t=args.t)
        print(p,r,f,m)
        prec.append(p)
        rec.append(r)
        fa.append(1-f)
        mAP.append(m)

    fig, ax1 = plt.subplots()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
   # ax1.plot(w_list,max_cr,'-o',markersize=6,label='Max-CR',linewidth = 3,color = colors[3])
    ax1.plot(w_list,cr_far,'-o',markersize=6,label='CR-far-patch',linewidth = 3,color = colors[2])
    ax1.plot(w_list,cr_close,'-o',markersize=6,label='CR-close-patch',linewidth = 3,color = colors[1])

    ax1.plot(w_list,cr_in,'-o',markersize=6,label='CR-over-patch',linewidth = 3,color = colors[0])
    ax1.set_yticks(ax1_y)
    ax1.grid()

    ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    if BOLD:
        ax1.set_xlabel('Window size',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    else:
        ax1.set_xlabel('Window size',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1

    print(w_list)
    print(mAP)
    print(fa)
    ax2.plot(w_list,np.array(mAP)*100,'-D',markersize=6,label='AP',linewidth = 3,color = colors[5])
    ax2.plot(w_list,np.array(fa)*100,'-D',markersize=6,label='1-FAR',linewidth = 3,color = colors[4])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,loc='lower right')
    ax2.set_yticks(ax2_y)
    ax2.set_xticks(np.arange(4,13,1))
    #ax.grid()
    #ax2.set_ylim(0, 35)
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig('w_voc.png')
    plt.savefig('w_voc.pdf')
    plt.close()

###########################################################################################

if args.para == 't':

    cr_in = []
    cr_close=[]
    cr_far=[]
    max_cr = []
    t_list = np.arange(28,36,1)
    mAP = []
    prec = []
    rec = []
    fa = []
    for t in t_list:
        cr = get_gt_pro('over',w=args.w,m=-1,p=8,num=args.num_img,t=t)
        cr_in.append(cr)
        cr = get_gt_pro('close',w=args.w,m=-1,p=8,num=args.num_img,t=t)
        cr_close.append(cr)
        cr = get_gt_pro('far',w=args.w,m=-1,p=8,num=args.num_img,t=t)
        cr_far.append(cr)
        
        p,r,f,m=get_gt_clean(w=args.w,m=-1,t=t)
        #print(p,r,f,m)
        prec.append(p)
        rec.append(r)
        fa.append(1-f)
        mAP.append(m)

    fig, ax1 = plt.subplots()
    #ax1.set_xlabel('Binarizing Threshold $T$')
    #ax1.set_ylabel('CR (%)')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #ax1.plot(t_list,max_cr,'-o',markersize=6,label='Max-CR',linewidth = 3,color = colors[3])

    ax1.plot(t_list,cr_in,'-o',markersize=6,label='CR-over-patch',linewidth = 3,color = colors[0])
    ax1.plot(t_list,cr_close,'-o',markersize=6,label='CR-close-patch',linewidth = 3,color = colors[1])
    ax1.plot(t_list,cr_far,'-o',markersize=6,label='CR-far-patch',linewidth = 3,color = colors[2])

    ax1.set_yticks(ax1_y)
    ax1.grid()


    ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    if BOLD:
        ax1.set_xlabel('Binarizing Threshold $T$',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    else:
        ax1.set_xlabel('Binarizing Threshold $T$',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    #ax2.plot(w_list,prec,label='prec')
    #ax2.plot(w_list,rec,label='rec')
    ax2.plot(t_list,np.array(mAP)*100,'-D',markersize=6,label='AP',linewidth = 3,color = colors[5])
    ax2.plot(t_list,np.array(fa)*100,'-D',markersize=6,label='1-FAR',linewidth = 3,color = colors[4])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,loc='lower right')
    ax2.set_yticks(ax2_y)

    #ax.grid()
    #ax2.set_ylim(0, 35)
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig('t_voc.png')
    plt.savefig('t_voc.pdf')
    plt.close()


#########################################################
if args.para == 'p':
    cr_in = []
    cr_close=[]
    cr_far=[]
    max_cr = []
    p_list = [5,6,7,8,9,10,11,12,13]
    #p_list = [6,7,8,9,10,11,12,13]
    mAP = []
    prec = []
    rec = []
    fa = []
    w=args.w
    for p in p_list:
        m=-1
        cr = get_gt_pro('over',w=w,m=m,p=p,num=args.num_img,t=args.t)
        cr_in.append(cr)
        cr = get_gt_pro('close',w=w,m=m,p=p,num=args.num_img,t=args.t)
        cr_close.append(cr)
        cr = get_gt_pro('far',w=w,m=m,p=p,num=args.num_img,t=args.t)
        cr_far.append(cr)
        #p,r,f,m=get_gt_clean(w=14,m=-1,t=t)
        #print(p,r,f,m)
        #prec.append(p)
        #rec.append(r)
        #fa.append(1-f)
        #mAP.append(m)
    print(cr_in)
    print(cr_close)
    print(cr_far)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Patch size (px)')
    ax1.set_ylabel('CR (%)')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    p_list = np.array(p_list)
    p_list = (p_list-8)*8 +32
    #ax1.plot(p_list,max_cr,'-o',markersize=6,label='Max-CR',linewidth = 3,color = colors[3])

    ax1.plot(p_list,cr_in,'-o',markersize=6,label='CR-over-patch',linewidth = 2,color = colors[0])
    ax1.plot(p_list,cr_close,'-o',markersize=6,label='CR-close-patch',linewidth = 2,color = colors[1])
    ax1.plot(p_list,cr_far,'-o',markersize=6,label='CR-far-patch',linewidth = 2,color = colors[2])

    #ax1.set_yticks(np.arange(0.0,0.62,0.1))
    ax1.grid()
    #ax1.set_xticks(np.arange(20,90,10))
    #ax1.set_yticks(np.arange(0,0.46,0.05))

    lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines , labels ,loc='upper right')
    #ax2.set_yticks(np.arange(0.90,1.01,0.01))

    #ax.grid()
    #ax2.set_ylim(0, 35)
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig('p_voc.png')
    plt.savefig('p_voc.pdf')
    plt.close()
if args.para == 'e':
    cr_in = []
    cr_close=[]
    cr_far=[]
    max_cr = []
    ems_list = [(1,4),(2,8),(3,24),(4,40),(5,67)]
    e_list = [1,2,3,4,5]
    mAP = []
    prec = []
    rec = []
    fa = []
    for e,ms in ems_list:
        cr = get_gt_pro('over',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,eps=e,ms=ms)
        cr_in.append(cr)
        cr = get_gt_pro('close',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,eps=e,ms=ms)
        cr_close.append(cr)
        cr = get_gt_pro('far',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,eps=e,ms=ms)
        cr_far.append(cr)
        
        p,r,f,m=get_gt_clean(w=args.w,m=-1,t=args.t,e=e,ms=ms)
        #print(p,r,f,m)
        prec.append(p)
        rec.append(r)
        fa.append(1-f)
        mAP.append(m)


    fig, ax1 = plt.subplots()
    ax1.set_xlabel('DBSCAN $\epsilon$')
    ax1.set_ylabel('CR (%)')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #ax1.plot(e_list,max_cr,'-o',markersize=6,label='Max-CR',linewidth = 3,color = colors[3])

    ax1.plot(e_list,cr_in,'-o',markersize=6,label='CR-over-patch',linewidth = 3,color = colors[0])
    ax1.plot(e_list,cr_close,'-o',markersize=6,label='CR-close-patch',linewidth = 3,color = colors[1])
    ax1.plot(e_list,cr_far,'-o',markersize=6,label='CR-far-patch',linewidth = 3,color = colors[2])

    ax1.set_yticks(ax1_y)
    ax1.grid()


    ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    if BOLD:
        ax1.set_xlabel('DBSCAN $\epsilon$',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    else:
        ax1.set_xlabel('DBSCAN $\epsilon$',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    ax2.plot(e_list,np.array(mAP)*100,'-D',markersize=6,label='AP',linewidth = 3,color = colors[5])
    ax2.plot(e_list,np.array(fa)*100,'-D',markersize=6,label='1-FAR',linewidth = 3,color = colors[4])


    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,loc='lower right')
    ax2.set_yticks(ax2_y)


    #ax2.set_ylim(0, 35)    #ax.grid()
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig('e_voc.png')
    plt.savefig('e_voc.pdf')
    plt.close()

if args.para == 'ms':
    cr_in = []
    cr_close=[]
    cr_far=[]
    max_cr = []
    ms_list = [14,16,18,20,22,24,26,28]
    mAP = []
    prec = []
    rec = []
    fa = []
    for ms in ms_list:
        cr = get_gt_pro('over',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,ms=ms)
        cr_in.append(cr)
        cr = get_gt_pro('close',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,ms=ms)
        cr_close.append(cr)
        cr = get_gt_pro('far',w=args.w,m=-1,p=8,num=args.num_img,t=args.t,ms=ms)
        cr_far.append(cr)
        
        p,r,f,m=get_gt_clean(w=args.w,m=-1,t=args.t,ms=ms)
        #print(p,r,f,m)
        prec.append(p)
        rec.append(r)
        fa.append(1-f)
        mAP.append(m)


    fig, ax1 = plt.subplots()
    #ax1.set_xlabel('DBSCAN min_points',weight='bold')
    #ax1.set_ylabel('CR (%)',weight='bold')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #ax1.plot(ms_list,max_cr,'-o',markersize=6,label='Max-CR',linewidth = 3,color = colors[3])

    ax1.plot(ms_list,cr_in,'-o',markersize=6,label='CR-over-patch',linewidth = 3,color = colors[0])
    ax1.plot(ms_list,cr_close,'-o',markersize=6,label='CR-close-patch',linewidth = 3,color = colors[1])
    ax1.plot(ms_list,cr_far,'-o',markersize=6,label='CR-far-patch',linewidth = 3,color = colors[2])

    ax1.set_yticks(ax1_y)
    ax1.set_xticks(np.arange(14,29,2))
    ax1.grid()


    ax2= ax1.twinx()  # instantiate a second axes that shares the same x-axis
    if BOLD:
        ax1.set_xlabel('DBSCAN min_points',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    else:
        ax1.set_xlabel('DBSCAN min_points',weight='bold')
        ax1.set_ylabel('CR (%)',weight='bold')    
        ax2.set_ylabel('AP/1-FAR (%)',weight='bold')  # we already handled the x-label with ax1
    ax2.plot(ms_list,np.array(mAP)*100,'-D',markersize=6,label='AP',linewidth = 3,color = colors[5])
    ax2.plot(ms_list,np.array(fa)*100,'-D',markersize=6,label='1-FAR',linewidth = 3,color = colors[4])


    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2,loc='lower right')
    ax2.set_yticks(ax2_y)
    #ax1.set_xticklabels(ax1.get_xticks(), font)
    #ax1.set_yticklabels(ax1.get_yticks(), font)
    #ax2.set_yticklabels(ax2.get_yticks(), font)
    #ax2.set_ylim(0, 35)    #ax.grid()
    #ax.set_ylim(-20,100)
    plt.tight_layout()
    plt.savefig('ms_voc.png')
    plt.savefig('ms_voc.pdf')
    plt.close()