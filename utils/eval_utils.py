##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
#from detectron vocevaluator

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""
import xml.etree.ElementTree as ET
import os
import numpy as np
import joblib 
from pycocotools.coco import COCO

def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with open(filename) as f:
        tree = ET.parse(f)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
       
        for t in np.arange(0.0, 1.01, 0.01):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / len(np.arange(0.0, 1.01, 0.01))
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def helper_voc(detpath,annopath,imagesetfile,fafile,classname):
    NUM_TEST_IMG = 4952  # hard code the size of test set of VOC
    # read gt for images in imagesetfile    
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # load gt annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    # read gt for images that has false alert (images in fafile)
    if fafile is not None:
        with open(fafile, "r") as f:
            lines = f.readlines()
        faimagenames = [x.strip() for x in lines]
        farecs = {}
        for imagename in faimagenames:
            farecs[imagename] = parse_rec(annopath.format(imagename))
        for imagename in faimagenames:
            R = [obj for obj in farecs[imagename] if obj["name"] == classname]
            difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
            npos = npos + sum(~difficult) # get the number objects in the fa images
        assert len(imagenames)+len(faimagenames) == NUM_TEST_IMG
    # read detection results of Base Detector
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines if x[0] in class_recs]
    
    confidence = np.array([float(x[1]) for x in splitlines if x[0] in class_recs])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines if x[0] in class_recs]).reshape(-1, 4)

    return NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB

def helper_coco(detpath,coco,imagesetfile,fafile,classname):
    NUM_TEST_IMG = 4952  # hard code the size of test set of COCO
    cat_ids = coco.getCatIds(catNms=[classname])
    if imagesetfile is None:
        imagenames = coco.getImgIds(catIds=cat_ids)
    else:
        with open(imagesetfile, "r") as f:
            lines = f.readlines()
        imagenames = [int(x.strip()) for x in lines]
    if fafile is not None:
        with open(fafile, "r") as f:
            lines = f.readlines()
        faimagenames = [int(x.strip()) for x in lines]
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        ann_ids = coco.getAnnIds(catIds=cat_ids,imgIds=[imagename])
        if len(ann_ids) ==0:
            continue
        R = coco.loadAnns(ann_ids)
        bbox = np.array([x["bbox"] for x in R])
        bbox[:,2:]+=bbox[:,:2]
        #difficult = np.zeros(len(bbox),dtype=bool)
        difficult = np.array([x["iscrowd"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
    #print(npos)

    if fafile is not None:
        for imagename in faimagenames:
            ann_ids = coco.getAnnIds(catIds=cat_ids,imgIds=[imagename])
            R = coco.loadAnns(ann_ids)
            #difficult = np.zeros(len(bbox),dtype=bool)
            difficult = np.array([x["iscrowd"] for x in R]).astype(np.bool)
            npos = npos + sum(~difficult)
        #print(len(imagenames),len(faimagenames))
        assert len(imagenames)+len(faimagenames) == NUM_TEST_IMG
    #print(npos)

    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [int(x[0]) for x in splitlines if int(x[0]) in class_recs]

    confidence = np.array([float(x[1]) for x in splitlines if int(x[0]) in class_recs])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines if int(x[0]) in class_recs]).reshape(-1, 4)

    return NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB



cls_map={'Car':0,'Van':0,'Truck':0,'Tram':0,'Pedestrian':1,'Person':1,'Cyclist':2}

def kitti_parse_rec(fn):
    with open(os.path.join(fn)) as rf:
        txt = rf.readlines()

    objs = []
    for line in txt:
        elements = line.split(' ')
        name = elements[0]
        if name not in cls_map:
            continue
        clss = cls_map[name]
        xmin = float(elements[4]) 
        ymin = float(elements[5]) 
        xmax = float(elements[6]) 
        ymax = float(elements[7]) 
        obj = {
            "bbox": [xmin,ymin,xmax,ymax],
            "name": clss,
        }
        objs.append(obj)
    return objs


def helper_kitti(detpath,annopath,imagesetfile,fafile,classname):
    NUM_TEST_IMG = 1496
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = kitti_parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    mmm = {'car':0,'person':1,'cyclist':2}
    classname = mmm[classname]
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        det = [False] * len(R)
        difficult = np.zeros(len(bbox),dtype=bool)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if fafile is not None:
        with open(fafile, "r") as f:
            lines = f.readlines()
        faimagenames = [x.strip() for x in lines]
        farecs = {}
        for imagename in faimagenames:
            farecs[imagename] = kitti_parse_rec(annopath.format(imagename))
        for imagename in faimagenames:
            R = [obj for obj in farecs[imagename] if obj["name"] == classname]
            npos = npos + len(R) # get the number objects in the fa images
        assert len(imagenames)+len(faimagenames) == NUM_TEST_IMG
       
    # read dets
    mmm = {0:'car',1:'person',2:'cyclist'}
    classname = mmm[classname]

    detfile = detpath.format(classname)

    with open(detfile, "r") as f:
        lines = f.readlines()
    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines if x[0] in recs]
    confidence = np.array([float(x[1]) for x in splitlines if x[0] in recs])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines if x[0] in recs]).reshape(-1, 4)
    return NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB

def eval_prec_rec(dataset,detpath, annopath, imagesetfile, fafile, classname, conf_thres, ovthresh=0.5):
    # calculate the precision and recall of DetectorGuard given a confidence threshold of Base Detector for one class
    '''
    INPUT:
    detpath             str, path to detections, detpath.format(classname) should produce the detection results file
    annopath            str, path to annotations, annopath.format(imagename) should be the xml annotations file
    imagesetfile        str, path to imagesetfile, which contains the image ids for which DetectorGuard does not alert (one line for one image id)
    fafile              str, path to imagesetfile, which contains the image ids for which DetectorGuard alerts (one line for one image id)
                        set to None if the evaluation is for undefended vanilla object detectors
    classname           str, classname for which precision and recall are calculated for
    conf_thres          float, the confidence threshold of Base DetectorGuard
    ovthresh            float, the IoU threshold, set to 0.5 to calcuate AP50

    OUTPUT:
    rec                 float, recall
    prec                float, precision
    fa                  float, false alert rate
    '''
    if dataset == 'voc':
        NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB = helper_voc(detpath,annopath,imagesetfile,fafile,classname)
    elif dataset == 'coco':
        coco = annopath
        NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB = helper_coco(detpath,coco,imagesetfile,fafile,classname)
    elif dataset == 'kitti':
        NUM_TEST_IMG,imagenames,class_recs,npos,image_ids,confidence,BB = helper_kitti(detpath,annopath,imagesetfile,fafile,classname)



    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    confidence = confidence[sorted_ind]
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # only retain detected bounding boxes whose prediction confidence is larger than the threshold
    fil_ind = np.where(confidence > conf_thres)[0]
    BB = BB[fil_ind, :]
    image_ids = [image_ids[x] for x in fil_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    if nd == 0: # no detection
        return 0,1,1-len(imagenames)/NUM_TEST_IMG

    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        if BBGT.size > 0:
            # compute overlaps
            # intersection
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
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            #tp[d] = 1.0
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.sum(fp)
    tp = np.sum(tp)
    #print(tp,fp,npos)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec, 1- len(imagenames)/NUM_TEST_IMG
