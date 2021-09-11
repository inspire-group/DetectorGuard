import torchvision
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import numpy as np 
import cv2
import random


NUM_CLASS_DICT = {'voc':20,'coco':80,'kitti':3}

FM_SIZE_DICT = {'voc':(48,48),'coco':(48,48),'kitti':(24,89)}

#####################################shared helper functions####################################################

# array([123.675, 116.28 , 103.53 ]) for RGB
# after normalization, this becomes [0,0,0]
def letterbox(img, new_shape=(416, 416), color=(103, 116, 123), auto=True, scaleFill=False, scaleup=True):
	#adapted from https://github.com/ultralytics/yolov3/issues/232
	#pad image to square and then resize  
	#input img hwc
	shape = img.shape[:2]  
	if isinstance(new_shape, int): 
		new_shape = (new_shape, new_shape)

	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) ## hw
	#print(r)
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) ## wh
	#print('new_unpad',new_unpad)
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		pass
		#dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = new_shape
		ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # width, height ratios ##???

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)


def collate_fn(batch):
	data = [item[0] for item in batch]
	target = [item[1] for item in batch]
	data = torch.stack(data)
	return data, target


#####################################PASACAL VOC####################################################
# customize torchvision.datasets.VOCDetection
# pad and resize images to 416x416

class VOCDetection(torchvision.datasets.VOCDetection):

	def __init__(self,root,year="2012",image_set="train",download=False,transforms=None):
		super().__init__(root,year=year,image_set=image_set,download=download)
		self._transforms = transforms
		self.training = 'train' in image_set
		self.classes_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor','bg']
		self.IMG_SIZE = 416
	def __getitem__(self, index):

		target = self.parse_voc_xml(
			ET.parse(self.annotations[index]).getroot())
		
		target = target['annotation']
		boxes = []
		labels = []

		for obj in target['object']:
			bb = obj['bndbox']
			bb = np.array([int(bb['xmin']),int(bb['ymin']),int(bb['xmax']),int(bb['ymax'])])-1 #subtract one such that coordinates start from zeros
			area = (bb[2]-bb[0]) * (bb[3]-bb[1])
			boxes.append(bb)
			labels.append(self.classes_name.index(obj['name']))

		img = cv2.imread(self.images[index])
		#pad to square and resize to (IMG_SIZE,IMG_SIZE)
		img, ratio, pad = letterbox(img, (self.IMG_SIZE,self.IMG_SIZE), auto=True)
		if len(boxes) > 0:
			labels = torch.tensor(labels)
			boxes = np.stack(boxes) 
			new_boxes = boxes.copy()
			#get scaled boxes
			new_boxes[:, 0] = ratio[0]  * (boxes[:, 0]) + pad[0]  # pad width
			new_boxes[:, 1] = ratio[1]  * (boxes[:, 1]) + pad[1]  # pad height
			new_boxes[:, 2] = ratio[0]  * (boxes[:, 2]) + pad[0]
			new_boxes[:, 3] = ratio[1]  * (boxes[:, 3]) + pad[1]
			if self.training and random.random() < 0.5: #horizontal flipping
				img = np.fliplr(img)
				a = IMG_SIZE - new_boxes[:, 0]
				b = IMG_SIZE - new_boxes[:, 2]
				new_boxes[:, 0] = b
				new_boxes[:, 2] = a
			new_target = {'bbox':new_boxes,'labels':labels,'img_id':target['filename'].split('.')[0],'ratio':ratio,'pad':pad}
		else:
			new_target = {'bbox':np.array([]),'labels':torch.tensor([]),'img_id':target['filename'].split('.')[0],'ratio':ratio,'pad':pad}
   
		img = img[:, :, ::-1]  # BGR to RGB
		img = np.ascontiguousarray(img)
		if self._transforms is not None:
			img = self._transforms(img)

		return img, new_target

#####################################MS COCO####################################################
# customize torchvision.datasets.VOCDetection
# pad and resize images to 416x416

import copy
import os

import torch.utils.data

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

import torchvision.transforms as T
import random


class CocoDetection(torchvision.datasets.CocoDetection):
	def __init__(self, img_folder, ann_file, transforms,training,img_size=416,fm_size=48,rf_size=33):
		super(CocoDetection, self).__init__(img_folder, ann_file)
		self._transforms = transforms
		self.training = training
		self.IMG_SIZE = 416
		self.coco_map_class = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 79, 80,None]

	def __getitem__(self, idx):
		img, target = super(CocoDetection, self).__getitem__(idx)
		image_id = self.ids[idx]
		target = dict(image_id=image_id, annotations=target)
		img, target = self.process_coco_data(img, target)
		img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

		img, ratio, pad = letterbox(img, (self.IMG_SIZE,self.IMG_SIZE), auto=True)

		boxes = target["bbox"]
		target['img_id']='{:012d}'.format(target['image_id'].item())
		if len(boxes) > 0:
			new_boxes = boxes.clone()
			new_boxes[:, 0] = ratio[0]  * (boxes[:, 0]) + pad[0]  # pad width
			new_boxes[:, 1] = ratio[1]  * (boxes[:, 1]) + pad[1]  # pad height
			new_boxes[:, 2] = ratio[0]  * (boxes[:, 2]) + pad[0]
			new_boxes[:, 3] = ratio[1]  * (boxes[:, 3]) + pad[1]
			if self.training and random.random() < 0.5:
				img = np.fliplr(img)
				a = IMG_SIZE - new_boxes[:, 0]
				b = IMG_SIZE - new_boxes[:, 2]
				new_boxes[:, 0] = b
				new_boxes[:, 2] = a
			target['bbox'] = new_boxes.numpy()
			target['ratio'] = ratio
			target['pad']=pad
		img = img[:, :, ::-1]  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img)
		if self._transforms is not None:
			img = self._transforms(img)
		return img, target

	#@staticmethod
	def process_coco_data(self, image, target):
		w, h = image.size
		image_id = target["image_id"]
		image_id = torch.tensor([image_id])
		anno = target["annotations"]
		anno = [obj for obj in anno if obj['iscrowd'] == 0] # remove iscrowd box
		boxes = [obj["bbox"] for obj in anno]
		# guard against no boxes via resizing
		boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
		boxes[:, 2:] += boxes[:, :2]
		boxes[:, 0::2].clamp_(min=0, max=w)
		boxes[:, 1::2].clamp_(min=0, max=h)
		classes = [self.coco_map_class[obj["category_id"]-1] for obj in anno]
		classes = torch.tensor(classes, dtype=torch.int64)
		keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
		boxes = boxes[keep]
		classes = classes[keep]
		target = {}
		target["bbox"] = boxes
		target["labels"] = classes
		target["image_id"] = image_id
		# for conversion to coco api
		area = torch.tensor([obj["area"] for obj in anno])
		iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
		target["area"] = area
		target["iscrowd"] = iscrowd

		return image, target
		
	@staticmethod
	def _coco_remove_images_without_annotations(dataset, cat_list=None):
		def _has_only_empty_bbox(anno):
			return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

		def _count_visible_keypoints(anno):
			return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

		min_keypoints_per_image = 10

		def _has_valid_annotation(anno):
			# if it's empty, there is no annotation
			if len(anno) == 0:
				return False
			# if all boxes have close to zero area, there is no annotation
			if _has_only_empty_bbox(anno):
				return False
			# keypoints task have a slight different critera for considering
			# if an annotation is valid
			if "keypoints" not in anno[0]:
				return True
			# for keypoint detection tasks, only consider valid images those
			# containing at least min_keypoints_per_image
			if _count_visible_keypoints(anno) >= min_keypoints_per_image:
				return True
			return False

		assert isinstance(dataset, torchvision.datasets.CocoDetection)
		ids = []
		for ds_idx, img_id in enumerate(dataset.ids):
			ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
			anno = dataset.coco.loadAnns(ann_ids)
			if cat_list:
				anno = [obj for obj in anno if obj["category_id"] in cat_list]
			if _has_valid_annotation(anno):
				ids.append(ds_idx)

		dataset = torch.utils.data.Subset(dataset, ids)
		return dataset


def get_coco(root, image_set, mode='instances'):
	anno_file_template = "{}_{}2017.json"
	PATHS = {
		"train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
		"val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
	}
	img_folder, ann_file = PATHS[image_set]
	img_folder = os.path.join(root,img_folder)
	ann_file = os.path.join(root, ann_file)
	transforms = T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
	training = image_set=='train' 
	dataset = CocoDetection(img_folder,ann_file,transforms=transforms,training=training)
	dataset = CocoDetection._coco_remove_images_without_annotations(dataset)
	return dataset


#####################################KITTI####################################################

# the size of kitti 2d images are almost the same, no padding is added for data preprocessing

class Kitti2DDetection(torch.utils.data.Dataset):

    def __init__(self,root,image_set="train"):
        
        with open(os.path.join(root,'{}.txt'.format(image_set))) as rf:
            id_list = rf.readlines()
        # merge into three classes
        # car, van, truck, tram
        # pedestrain, person
        # cyclist

        cls_map={'Car':0,'Van':0,'Truck':0,'Tram':0,'Pedestrian':1,'Person':1,'Cyclist':2}
        self.IMG_SIZE = (740,224)

        annotations = []
        images = []
        for img_id in id_list:
            img_id = img_id.strip()
            img_path = os.path.join(root,'image_2','{}.png'.format(img_id))
            img = Image.open(img_path)
            w_ratio = self.IMG_SIZE[0] / img.size[0]
            h_ratio = self.IMG_SIZE[1] / img.size[1]
            images.append(img_path)
            target = {}
            target['img_id'] = img_id
            with open(os.path.join(root,'label_2','{}.txt'.format(img_id))) as rf:
                txt = rf.readlines()
            bbox_list = []
            label_list = []
            for line in txt:
                elements = line.split(' ')
                name = elements[0]
                if name not in cls_map:
                    continue
                clss = cls_map[name]
                xmin = float(elements[4]) * w_ratio #w
                ymin = float(elements[5]) * h_ratio #h
                xmax = float(elements[6]) * w_ratio
                ymax = float(elements[7]) * h_ratio
                label_list.append(clss)
                bbox_list.append(np.array([xmin,ymin,xmax,ymax]))
            if len(label_list)==0:
                continue
            label_list = torch.tensor(label_list)
            bbox_list = np.stack(bbox_list)
            target = {'img_id':img_id,'labels':label_list,'bbox':bbox_list,'ratio':[w_ratio,h_ratio],'pad':[0,0]}
            annotations.append(target)

        self.annotations = annotations
        self.images = images
        self._transforms = T.Compose([T.Resize((self.IMG_SIZE[1],self.IMG_SIZE[0])),T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        target = self.annotations[index]
        img = Image.open(self.images[index])
        img = self._transforms(img)
        return img, target

    def __len__(self):
        return len(self.annotations)
