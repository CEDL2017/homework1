import os
import numpy as np
from shutil import copyfile
import ipdb

train_dir = "/home/johnson/Desktop/CEDL/homework1/data/train"
test_dir = "/home/johnson/Desktop/CEDL/homework1/data/test"
labels_root_dir = "/home/johnson/Desktop/CEDL/homework1/data/labels"

Obj = { 'free':0,
        'computer':1,
        'cellphone':2,
        'coin':3,
        'ruler':4,
        'thermos-bottle':5,
        'whiteboard-pen':6,
        'whiteboard-eraser':7,
        'pen':8,
        'cup':9,
        'remote-control-TV':10,
        'remote-control-AC':11,
        'switch':12,
        'windows':13,
        'fridge':14,
        'cupboard':15,
        'water-tap':16,
        'toy':17,
        'kettle':18,
        'bottle':19,
        'cookie':20,
        'book':21,
        'magnet':22,
        'lamp-switch':23 }

#places = ['house','office','lab']
places = ['lab']
nums = ['1','2','3','4']
parts = ['head','Lhand','Rhand']
label_parts = ['head','left','right']
'''
# make directories for new train dataset
root_dir = "/home/johnson/Desktop/CEDL/homework1/data/reformed_train"
if not os.path.isdir(root_dir):
    os.mkdir(root_dir) # root directory
for subdir in Obj.keys():
    dir_path = os.path.join(root_dir,subdir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        print("{} already exists, did not create a new one".format(dir_path))

# train set 
for place in places:
    for num in nums:
        if num=='4' and place!='lab': # only lab has 4
            continue

        print("Processing {}/{}/{}...".format(train_dir,place,num))

        # extract label
        labels_dir = os.path.join(labels_root_dir,place)
        labels_l = np.load(os.path.join(labels_dir,"obj_left{}.npy".format(num)))
        labels_r = np.load(os.path.join(labels_dir,"obj_right{}.npy".format(num)))

        for part in parts:
            # extract image file name
            img_dir = os.path.join(train_dir,place,num,part)
            img_file_list = os.listdir(img_dir)

            if part=="Lhand": # left hand
                # copy images to construct reformed dataset according to label_l
                for idx, img_fname in enumerate(img_file_list):
                    img_fname_no_ext, fname_ext = img_fname.split(".")
                    if fname_ext=="png":
                        img_idx = int(img_fname_no_ext[5:])
                        img_label = int(labels_l[img_idx-1]) # number label
                        img_slabel = Obj.keys()[Obj.values().index(img_label)] # semantic label
                        old_img_path = os.path.join(img_dir,img_fname)
                        new_img_path = os.path.join(root_dir,img_slabel,"{}_{}_{}_{}".format(place,num,part,img_fname))
                        copyfile(old_img_path,new_img_path)
                    #if idx>5:#DEBUG
                    #    break

            elif part=="Rhand": # right hand
                # copy images to construct reformed dataset according to label_l
                for idx, img_fname in enumerate(img_file_list):
                    img_fname_no_ext, fname_ext = img_fname.split(".")
                    if fname_ext=="png":
                        img_idx = int(img_fname_no_ext[5:])
                        img_label = int(labels_r[img_idx-1]) # number label
                        img_slabel = Obj.keys()[Obj.values().index(img_label)] # semantic label
                        new_img_path = os.path.join(root_dir,img_slabel,"{}_{}_{}_{}".format(place,num,part,img_fname))
                        copyfile(old_img_path,new_img_path)
                    #if idx>5:#DEBUG
                    #    break
'''
# make directories for new test dataset
root_dir = "/home/johnson/Desktop/CEDL/homework1/data/reformed_test"
if not os.path.isdir(root_dir):
    os.mkdir(root_dir) # root directory
for subdir in Obj.keys():
    dir_path = os.path.join(root_dir,subdir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    else:
        print("{} already exists, did not create a new one".format(dir_path))

# test set 
for place in places:
    for num in nums:
        label_num = int(num) + 3
        if num=='4' and place!='lab': # only lab has 4
            continue
        elif place=='lab':
            label_num = int(num) + 4

        print("Processing {}/{}/{}".format(test_dir,place,num))

        # extract label
        labels_dir = os.path.join(labels_root_dir,place)
        labels_l = np.load(os.path.join(labels_dir,"obj_left{}.npy".format(label_num)))
        labels_r = np.load(os.path.join(labels_dir,"obj_right{}.npy".format(label_num)))

        for part in parts:
            # extract image file name
            img_dir = os.path.join(test_dir,place,num,part)
            img_file_list = os.listdir(img_dir)

            if part=="Lhand": # left hand
                # copy images to construct reformed dataset according to label_l
                for idx, img_fname in enumerate(img_file_list):
                    img_fname_no_ext, fname_ext = img_fname.split(".")
                    if fname_ext=="png":
                        img_idx = int(img_fname_no_ext[5:])
                        img_label = int(labels_l[img_idx-1]) # number label
                        img_slabel = Obj.keys()[Obj.values().index(img_label)] # semantic label
                        old_img_path = os.path.join(img_dir,img_fname)
                        new_img_path = os.path.join(root_dir,img_slabel,"{}_{}_{}_{}".format(place,num,part,img_fname))
                        copyfile(old_img_path,new_img_path)
                    #if idx>5:#DEBUG
                    #    break

            elif part=="Rhand": # right hand
                # copy images to construct reformed dataset according to label_l
                for idx, img_fname in enumerate(img_file_list):
                    img_fname_no_ext, fname_ext = img_fname.split(".")
                    if fname_ext=="png":
                        img_idx = int(img_fname_no_ext[5:])
                        img_label = int(labels_r[img_idx-1]) # number label
                        img_slabel = Obj.keys()[Obj.values().index(img_label)] # semantic label
                        new_img_path = os.path.join(root_dir,img_slabel,"{}_{}_{}_{}".format(place,num,part,img_fname))
                        copyfile(old_img_path,new_img_path)
                    #if idx>5:#DEBUG
                    #    break

