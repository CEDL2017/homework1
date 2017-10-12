#############################################################################################################
# load images and labels                                                                                    #
# image dimension : (N,h,w,ch), label dimension : (N,)                                                      #
# type of all the output will be "np.array"                                                                 #
#############################################################################################################
# There are two different setting of training and testing :                                                 #
# The default setting (setting-0) is splitting the dataset into two parts, half for training, half for      #
# testing. Training index: lab-1234, office-123, house-123. Testing index: lab-5678, office-456, house-456  #
#############################################################################################################

import sys, os, re, io, math
import numpy as np
import cv2
import warnings

def load_one_label_seq(zip_ref, path):
	npy = zip_ref.read(path)
	npy = io.BytesIO(npy)
	npy = np.load(npy)
	return npy.astype(int)


def load_label_seqs(zip_ref, mode, index, for_head):
	labels = []
	l_labels = []
	r_labels = []
	for i in range(len(index)):
		loc = index[i][0]
		idx = index[i][1]
		left_labelnpy = 'labels/'+loc+'/'+mode+'_left'+str(idx)+'.npy'
		left_labels = load_one_label_seq(zip_ref, left_labelnpy)
		right_labelnpy = 'labels/'+loc+'/'+mode+'_right'+str(idx)+'.npy'
		right_labels = load_one_label_seq(zip_ref, right_labelnpy)
		if for_head:
			l_labels.extend(left_labels)
			l_labels.extend(left_labels)
			r_labels.extend(right_labels)
			r_labels.extend(right_labels)
		else:
			labels.extend(left_labels)
			labels.extend(right_labels)
	if for_head:
		labels.append(l_labels)
		labels.append(r_labels)
	return labels

def gen_label_index(setting_index):
	train_index = []
	test_index = []
	if setting_index == 0:
		for i in range(1,9):
			if i <= 4:
				train_index.append(('lab',i))
			else:
				test_index.append(('lab',i))
		for i in range(1,7):
			if i <= 3:
				train_index.append(('office',i))
			else:
				test_index.append(('office',i))
		for i in range(1,7):
			if i <= 3:
				train_index.append(('house',i))
			else:
				test_index.append(('house',i))
	elif setting_index == 1:
		for i in range(1,9):
			train_index.append(('lab',i))
		for i in range(1,7):
			train_index.append(('office',i))
		for i in range(1,7):
			test_index.append(('house',i))
	else:
		raise ValueError('error setting index')
	return train_index, test_index



def gen_label_index_process(index=None, setting_index=None):
	if index == None:
		if setting_index == None:
			raise ValueError('Setting index can not be none')
		else:
			train_index, test_index = gen_label_index(setting_index)
	return train_index, test_index


def load_train_labels(zip_ref, mode, index=None, setting_index=None, for_head=False):
	if index == None:
		index, _ = gen_label_index_process(index, setting_index)
	else:
		if setting_index != None:
			warnings.warn('setting_index has no effect when given particular index')
	labels = load_label_seqs(zip_ref, mode, index, for_head)
	return np.array(labels)

def load_test_labels(zip_ref, mode, index=None, setting_index=None, for_head=False):
	if index == None:
		_, index = gen_label_index_process(index, setting_index)
	else:
		if setting_index != None:
			warnings.warn('setting_index has no effect when given particular index')
	labels = load_label_seqs(zip_ref, mode, index, for_head)
	return np.array(labels)


def load_all_labels(zip_ref, mode, setting_index, for_head=False):
	train_index, test_index = gen_label_index(setting_index)
	train_labels = load_train_labels(zip_ref, mode, train_index, for_head)
	test_labels = load_test_labels(zip_ref, mode, test_index, for_head)
	return np.array(train_labels), np.array(test_labels)





def get_image_name_array(zip_ref, dataset_name): # dataset_name = ['train', 'test']: files for each set in zip
	name_list = sorted([name for name in zip_ref.namelist() if '.' in os.path.split(name)[1]]) # if file
	hand_image_name_list = []
	head_image_name_list = []
	for loc in [('lab', 1), ('lab', 2), ('lab', 3), ('lab', 4), ('office', 1), ('office', 2), ('office', 3), ('house', 1), ('house', 2), ('house', 3)]:
		for cam_loc in ['head', 'Lhand', 'Rhand']:
			sub_name = [name for name in name_list if (dataset_name+'/'+loc[0]+'/'+str(loc[1])+'/'+cam_loc) in name]
			sub_name_sorted = sorted(sub_name, key=lambda x: int(re.sub('\D', '', os.path.split(x)[1]))) # change file name to numerical order
			if cam_loc == 'head': # duplicate head
				head_image_name_list.extend(sub_name_sorted + sub_name_sorted)
			else: # Lhand + Rhand
				hand_image_name_list.extend(sub_name_sorted)
	return np.array(hand_image_name_list), np.array(head_image_name_list)


def load_image(zip_ref, path, img_height, img_width, to_random_crop=True, to_flip=False): # including pre-processing
	image = zip_ref.read(path)
	nparr = np.fromstring(image, np.uint8)
	image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	image = image.astype(np.single)
	if to_random_crop:
		resized_height = 2 ** int(math.ceil(math.log(img_height, 2)))
		resized_width = 2 ** int(math.ceil(math.log(img_width, 2)))
		image = cv2.resize(image, (resized_height,resized_width)) # imresize
		# remove image mean
		image[:,:,0] -= 103.939
		image[:,:,1] -= 116.779
		image[:,:,2] -= 123.68
		rand_height = np.random.random_integers(resized_height-img_height) # random height
		rand_width = np.random.random_integers(resized_width-img_width) # random width
		image = image[rand_height:(rand_height+img_height), rand_width:(rand_width+img_width), :]
	else:
		image = cv2.resize(image, (img_height,img_width)) # directly imresize
	if to_flip: # horizontal flip
		image = image[:, ::-1, :]
	image = image/255 # im2double, normalize to [0,1]
	return image

def load_images(zip_ref, img_height, img_width, image_name_array=[], to_random_crop=True):
	images = []
	for image_path in image_name_array:
		to_flip = True if 'Lhand' in image_path else False # flip left hand to mimic right hand
		image = load_image(zip_ref, image_path, img_height, img_width, to_random_crop=to_random_crop, to_flip=to_flip)
		images.append(image)
	return np.array(images)
