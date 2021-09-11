import os
import numpy as np 

def bboxes_img2fm(img_bboxes,FM_SIZE,RF_SIZE,RF_STRIDE=8):
    # map pixel-space bounding boxes to feature-space coordinates
    '''
    INPUT:
    img_bboxes      np.ndarray, [N,4], bounding boxes in pixel coordinates 
    FM_SIZE         tuple, (int,int), the size of feature map
    RF_SIZE         int, the size of receptive field
    RF_STRIDE       int, the stride of receptive field (8 for the BagNet implementation used in this paper)

    OUTPUT:
    fm_bboxes       np.ndarray, [N,4], bounding boxes in feature coordinates 
    '''

    img_bboxes = img_bboxes.copy()
    img_bboxes[:,:2] = img_bboxes[:,:2] - RF_SIZE + 1 
    fm_bboxes = img_bboxes / RF_STRIDE
    fm_bboxes = np.floor(fm_bboxes)
    fm_bboxes = fm_bboxes[:,[1,0,3,2]] 
    if isinstance(FM_SIZE,int):
        fm_bboxes = np.clip(fm_bboxes,0,FM_SIZE)
    else:
        fm_bboxes[:,2] = np.clip(fm_bboxes[:,2],0,FM_SIZE[0])
        fm_bboxes[:,3] = np.clip(fm_bboxes[:,3],0,FM_SIZE[1])
        fm_bboxes[:,0] = np.clip(fm_bboxes[:,0],0,FM_SIZE[0])
        fm_bboxes[:,1] = np.clip(fm_bboxes[:,1],0,FM_SIZE[1])

    return fm_bboxes.astype(int)

def rescale_det(det,ratio,pad):
    # rescale prediction bounding boxes for the padded and re-sclaced 416x416 images
    '''
    INPUT:
    det             np.ndarray, [N,6] for N objects, each object has [pred_cls, conf, x_min,y_min,x_max,y_max]
    ratio           float, the rescaling ratio used in image pre-processing (when transforming the image to 416x416)
    pad             tuple, (int,int), the number of pixels padded in the image pre-processing (when transforming the image to 416x416)

    OUTPUT:
    img_bboxes      np.ndarray, [N,4], rescaled boxes in pixel coordinates 
    confs           np.ndarray, [N], prediction confidence for each bounding box
    labels          np.ndarray, [N], prediction class label for each bounding box
    '''

    labels = det[:,0]
    confs =  det[:,1]
    bboxes = det[:,2:]
    img_bboxes = bboxes.copy()
    img_bboxes[:, 0] = ratio[0]  * (bboxes[:, 0]) + pad[0]  # pad width
    img_bboxes[:, 1] = ratio[1]  * (bboxes[:, 1]) + pad[1]  # pad height
    img_bboxes[:, 2] = ratio[0]  * (bboxes[:, 2]) + pad[0]
    img_bboxes[:, 3] = ratio[1]  * (bboxes[:, 3]) + pad[1]  

    return img_bboxes,confs,labels

def read_det(input_dir):
    # read Base Detector predictions from input directory
    '''
    INPUT:
    input_dir       str, the directory for base detector predictions

    OUTPUT:
    det_raw         dict {img_id:[[pred_cls, conf, x_min,y_min,x_max,y_max],...]}, a dict with predictions for each image
    '''

    det_raw = {}
    fn_list = os.listdir(input_dir)
    print('reading detection file from {}'.format(input_dir))
    for fn in fn_list:
        with open(input_dir+'/'+fn,'r') as rf:
            lines = rf.readlines()
        splitlines = [x.strip().split(" ") for x in lines]
        tmp=[]
        for line in splitlines:
            tmp.append(np.array([float(x) for x in line])) # pred_cls, conf, xyxy
        tmp = np.stack(tmp)
        img_id = fn[:-4]
        det_raw[img_id] =tmp

    return det_raw


def get_box(BBGT,bb,confs,ovthresh=0.5):
    confs=confs.copy()
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
    ih = np.maximum(iymax - iymin + 1.0, 0.0)
    inters = iw * ih

    # union
    uni = (
        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
        + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
        - inters
    )

    overlaps = inters / uni

    ovmax = np.max(overlaps)
    
    tmp = overlaps > ovthresh
    confs = confs[tmp]
    if len(confs)==0:
        return -2.
    else:
        return np.max(confs)

'''
def checkoverlapping(xyxy1,xyxy2):#unused
    return (xyxy1[0] <= xyxy2[2] and xyxy2[0]<=xyxy1[2]) and (xyxy1[1] <= xyxy2[3] and xyxy2[1]<=xyxy1[3])
'''