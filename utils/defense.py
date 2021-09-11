import numpy as np 
from scipy.special import softmax
from scipy import ndimage

def sliding_window_mean(local_feature,window_size):

	#helper function, calculate the mean of each sliding window

	'''
	INPUT:
	local_features			np.ndarray, [W,H,num_cls], the local feature map (i.e., local logits map)
	window_size 			(int,int) or int, the size of sliding window

	far_listPUT:
	in_mask_mean_tensor		np.ndarray, [W',H',num_cls], the tensor that contains the mean vector of every window

	'''

	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_size if not isinstance(window_size,int) else (window_size,window_size)
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1
	# use ndimage.uniform_filter 
	in_window_mean_tensor = ndimage.uniform_filter(local_feature, size=(window_size_x,window_size_y,1),mode='constant',cval=0,origin=(-(window_size_x//2),-(window_size_y//2),0))[:num_window_x,:num_window_y]
	#in_window_sum_tensor *= window_size_x*window_size_y

	# a less efficient but more intuitive implemtation
	#in_window_sum_tensor = np.zeros([num_window_x,num_window_y,num_cls])
	#for x in range(num_window_x):
	#	for y in range(num_window_y):
	#		in_window_sum_tensor[x,y,:] = np.sum(local_feature_window[x:x+window_size_x,y:y+window_size_y],axis=(0,1))
	return in_window_mean_tensor



def clipping_(local_feature_window,window_mean,patch_box=[]):
	# helper function for clipping-based Objectness Predictor
	# secure aggregation/prediction for one feature-sapce window
	# if patch_box is [], perform clipping aggregation (in the clean setting), return the aggregated vector
	# if patch_box is not empty, perform provable analysis for this patch box, return the lower bound of aggregated vector

	'''
	INPUT:
	local_feature_window 			np.ndarray, [window_size_x,window_size_y,num_cls], a window of local feature map
	window_mean 					np.ndarray, [num_cls], the mean vector of this window of local features
	patch_box 						list of np.ndarray, a list of patch location coordinates

	far_listPUT:
	aggregated_feature				np.ndarray, [num_cls], the aggregated mean vec (in the clean setting), or the lower bound of the vector (when patch_box is non-empty)

	'''
	window_size_x,window_size_y,num_cls = local_feature_window.shape
	patch_sum = 0
	for pb in patch_box:
		tmp = np.sum(local_feature_window[pb[0]:pb[2],pb[1]:pb[3]],axis=(0,1))
		patch_sum = patch_sum+tmp if np.sum(patch_sum)>0 else tmp
	aggregated_feature = window_mean - patch_sum/(window_size_x*window_size_y)
	return aggregated_feature


def masking_(local_feature_window,window_sum,mask_size,in_mask_sum_tensor,patch_box=[]):
	# helper function for masking-based Objectness Predictor (from PatchGuard's robust masking)
	# secure aggregation/prediction for one feature-sapce window
	# if patch_box is [], perform robust aggregation (in the clean setting), return the aggregated vector
	# if patch_box is not empty, perform provable analysis for patch boxes, return the lower bound of aggregated vector
	# adapted from https://github.com/inspire-group/PatchGuard/blob/master/utils/defense_utils.py
	'''
	INPUT:
	local_feature_window 			np.ndarray, [window_size_x,window_size_y,num_cls], a window of local feature map
	window_sum 						np.ndarray, [num_cls], the sum vector of this window of local features
	in_mask_sum_tensor				np.ndarray, the tensor obatined from sliding_window_mean() with the mask size as the input
	patch_box 						list of np.ndarray, a list of patch location coordinates

	far_listPUT:
	aggregated_feature				np.ndarray, [num_cls], the aggregated mean vec (in the clean setting), or the lower bound of the vector (when patch_box is non-empty)

	'''
	window_size_x,window_size_y,num_cls = local_feature_window.shape
	num_mask_x = window_size_x - mask_size + 1
	num_mask_y = window_size_y - mask_size + 1
	mask_size_x = mask_size_y = mask_size

	assert len(patch_box)<=1 # does not support multiple patches for now
	if len(patch_box)==1:
		patch_box = patch_box[0]
	
	if len(patch_box)==0 or (patch_box[2]-patch_box[0])*(patch_box[3]-patch_box[1])==0: # no patch, or the patch is not presented in this window --> clean setting
		max_in_mask_sum = np.max(in_mask_sum_tensor,axis=(0,1))
		window_logits = window_sum - max_in_mask_sum
		return window_logits/(window_size_x*window_size_y - mask_size ** 2)
	else:
		assert max(patch_box[2]-patch_box[0],patch_box[3]-patch_box[1])<=mask_size # mask size should be larger than patch size
		#lower bound
		in_mask_sum_pred_patched = in_mask_sum_tensor.copy()
		# only need to recalculate the windows the are partially patched
		for xx in range(max(0,patch_box[0] - mask_size + 1),min(patch_box[2],num_mask_x)):
			for yy in range(max(0,patch_box[1] - mask_size+ 1),min(patch_box[3],num_mask_y)):
				#in_mask_sum_pred_patched[xx,yy,:]=np.sum(local_feature_window_patched[xx:xx+mask_size_x,yy:yy+mask_size],axis=(0,1))
				in_mask_sum_pred_patched[xx,yy,:]-=np.sum(local_feature_window[max(patch_box[0],xx):min(patch_box[2],xx+mask_size),max(patch_box[1],yy):min(patch_box[3],yy+mask_size)],axis=(0,1))
		max_mask_sum_pred = np.max(in_mask_sum_pred_patched,axis=(0,1))
		global_feature_patched= window_sum - np.sum(local_feature_window[patch_box[0]:patch_box[2],patch_box[1]:patch_box[3]],axis=(0,1)) - max_mask_sum_pred

		return global_feature_patched/(window_size_x*window_size_y - mask_size ** 2)


def gen_obj_map(local_features,window_size,pad,mode,patch_box_abs=[],mask_size=None):
	# main function of Objectness Predictor: perform robust classification over a feature-space sliding window
	# if patch_box_abs is [], perform robust aggregation (in the clean setting), return the objectness map
	# if patch_box is not empty, perform provable analysis for patch boxes, return the lower bound of the objectness map

	'''
	INPUT:
	local_features		 			np.ndarray, local feature map extracted from BagNet
	window_size 					(int,int) or int, the size of sliding window
	pad 							(int,int), the number of pixels padded when transforming the image to 416x416 (not essential to the core function though)
	mode							str, 'clip' or 'mask', the type of secure aggregation
	patch_box_abs					list of np.ndarray, a list of patch (feature-space) coordinates
	mask_size 						int, the mask size (used when mode=='mask')

	far_listPUT:
	obj_map							np.ndarray, objectness map withfar_list binarization

	'''
	## this part is to zero far_list local features that see part of padded pixels (not necessary, but found helpful for reducing false alerts)
	pad = [int(x//8) for x in pad]
	if pad[1]>0:
		pad[1]+=1
		local_features[:pad[1]]=0
		local_features[-pad[1]:]=0
	elif pad[0]>0:
		pad[0]+=1
		local_features[:,:pad[0]]=0
		local_features[:,-pad[0]:]=0

	local_features = np.clip(local_features,0,np.inf)
	FM_SIZE = (local_features.shape[0],local_features.shape[1])
	num_cls = local_features.shape[-1]
	num_window_x = FM_SIZE[0] - window_size +1
	num_window_y = FM_SIZE[1] - window_size +1

	obj_map = np.zeros([FM_SIZE[0],FM_SIZE[1],num_cls])

	if mode == 'clip':
		window_mean_tensor = sliding_window_mean(local_features,window_size)
		for x in range(num_window_x):
			for y in range(num_window_y):
				# note that the patch_box passed to clipping_() should be relative coordinates for that window
				patch_box = [np.clip(pb_abs-np.array([x,y,x,y]),0,window_size) for pb_abs in patch_box_abs]
				window_logits = clipping_(local_features[x:x+window_size,y:y+window_size],window_mean_tensor[x,y],patch_box=patch_box)
				obj_map[x:x+window_size,y:y+window_size,:] = obj_map[x:x+window_size,y:y+window_size,:] + window_logits
	elif mode =='mask':
		num_mask = window_size - mask_size +1
		window_sum_tensor = sliding_window_mean(local_features,window_size) * window_size**2
		mask_sum_tensor = sliding_window_mean(local_features,mask_size) * mask_size**2
		for x in range(num_window_x):
			for y in range(num_window_y):
				# note that the patch_box passed to clipping_() should be relative coordinates for that window
				patch_box = [np.clip(pb_abs-np.array([x,y,x,y]),0,window_size) for pb_abs in patch_box_abs]
				window_logits = masking_(local_features[x:x+window_size,y:y+window_size],window_sum_tensor[x,y],mask_size,mask_sum_tensor[x:x+num_mask,y:y+num_mask],patch_box=patch_box)
				obj_map[x:x+window_size,y:y+window_size,:] = obj_map[x:x+window_size,y:y+window_size,:] + window_logits
	obj_map = np.max(obj_map[:,:,:-1],axis=-1) #discard the ``background`` class


	## this part is to zero far_list local features that see part of padded pixels (not necessary, but found helpful for reducing false alerts)
	if pad[1]>0:
		pad[1]+=1
		obj_map[:pad[1]]=0
		obj_map[-pad[1]:]=0
	elif pad[0]>0:
		pad[0]+=1
		obj_map[:,:pad[0]]=0
		obj_map[:,-pad[0]:]=0

	return obj_map


def explainer(fm_bboxes,obj_map,dbscan,remove_offset=0):
	# perform objectness explaining
	'''
	INPUT:
	fm_bboxes 			np.ndarray, [N,4], feature-space bounding box coordinates from Base Detector
	obj_map 			np.ndarray, objectness map from Objectness Predictor
	dbscan 				the sklearn.cluster.DBSCAN instance for DBSCAN clustering
	remove_offset		int, a parameter indicating how aggressive the explaining is (not used in the paper)

	far_listPUT:
	fn_flgs				np.ndarray (bool), an array indicating whether there is a false negative of Objectness Predictor for each object
	alert_flg			bool, whether we will issue an alert for this image 
	'''
	num_bboxes = len(fm_bboxes)
	fm_size_x,fm_size_y = obj_map.shape
	obj_map_copy = obj_map.copy()
	fn_flgs = np.zeros((num_bboxes),dtype=bool)

	# use each bbox to explain objectenss in obj_map
	for i in range(num_bboxes):
		x_min,y_min,x_max,y_max = fm_bboxes[i]
		if np.sum(obj_map[x_min:x_max,y_min:y_max]) < 1: #false negative of Objectness Predictor --> benign mismatch
			fn_flgs[i]=True 
		else: # match, explaining (zeroing far_list) objectness 
			x_min = max(x_min - remove_offset,0)
			y_min = max(y_min - remove_offset,0)
			x_max = min(x_max + remove_offset,fm_size_x)
			y_max = min(y_max + remove_offset,fm_size_y)
			obj_map_copy[x_min:x_max,y_min:y_max]=0

	# check unexplained objectness
	alert_flg = False
	coords = np.stack(np.where(obj_map_copy>0)).T
	if len(coords)>0:
		cluster_labels = dbscan.fit_predict(coords)
		num_clusters = np.max(cluster_labels)+1
		if num_clusters > 0:
			alert_flg = True
	return fn_flgs,alert_flg


def gen_patch_loc(fm_box,patch_size,FM_SIZE,pskip=1,onoffset=1):
	# generate a list of locations for a given object bounding box (in terms of over-patch, close-patch, far-patch threat models)

	'''
	INPUT:
	fm_bbox 			np.ndarray, [4], feature-space bounding box coordinates the object 
	patch_size 			(int,int), patch size
	FM_SIZE 			(int,int), feature map size
	pskip				int, the stride of patch location (should be 1 for provable anlaysis)
	onoffset			int, a parameter tunning the boundary between different threat models

	OUTPUT:
	{'over':over_list,'close':close_list,'far':far_list}
	a dict for patch locations within different threat models
	'''
	over_list = []
	close_list = []
	far_list = []
	for patch_x in range(0,FM_SIZE[0]-patch_size[0]+1,pskip):
		for patch_y in range(0,FM_SIZE[1]-patch_size[1]+1,pskip):
			if fm_box[0]-patch_size[0]//2 <= patch_x < fm_box[2]-patch_size[0]//2+1 and fm_box[1]-patch_size[1]//2 <= patch_y < fm_box[3]-patch_size[1]//2+1:
				over_list.append((patch_x,patch_y))
			elif fm_box[0] - patch_size[0] - onoffset <= patch_x < fm_box[2] +1 + onoffset and fm_box[1] -patch_size[1]-onoffset<= patch_y < fm_box[3] + onoffset +1:
				close_list.append((patch_x,patch_y))
			else:
				far_list.append((patch_x,patch_y))
	return {'over':over_list,'close':close_list,'far':far_list}


def check_vul(raw_obj_map,thres,fm_box,dbscan,WINDOW_SIZE):
	# check if this object (bounding box) is possibly vulnerable
	# i.e., check if there is high objctness in the worst-case objectness map
	x_min,y_min,x_max,y_max = fm_box
	obj_interested = raw_obj_map[x_min:x_max,y_min:y_max] > WINDOW_SIZE**2 *thres
	coords = np.stack(np.where(obj_interested)).T
	if len(coords)>0:
		cluster_labels = dbscan.fit_predict(coords)
		num_clusters = np.max(cluster_labels)+1
	else:
		num_clusters = 0
	return num_clusters == 0

'''
def get_filtered(obj_map,dbscan):
	# perform DBSCAN to first filter out outliers
	obj_map = obj_map.copy()
	coords = np.stack(np.where(obj_map>0)).T
	if len(coords)>0:
		cluster_labels = dbscan.fit_predict(coords)##########################	
		for coor,ll in zip(coords,cluster_labels):
			if ll==-1:
				obj_map[coor[0],coor[1]] =0 
	return obj_map
'''