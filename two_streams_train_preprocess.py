import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
FOLDER_DATASET = "data/frames"
from glob import glob
import pickle

# print(glob("data/frames/train/house/1/Lhand/*.png"))
train_dataset_list = []
train_X_list = []
train_y_list = []
#*****************************************************************************************************************
############################################################################################################ house
for num in range(1, 3+1):
    img_list = glob("data/frames/train/house/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/house/obj_left{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/house/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/house/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("house LHand done!")

for num in range(1, 3+1):
    img_list = glob("data/frames/train/house/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/house/obj_right{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/house/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/house/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("house RHand done!")
############################################################################################################
############################################################################################################ lab
for num in range(1, 4+1):
    img_list = glob("data/frames/train/lab/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/lab/obj_left{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/lab/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/lab/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("Lab LHand done!")

for num in range(1, 4+1):
    img_list = glob("data/frames/train/lab/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/lab/obj_right{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/lab/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/lab/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("lab RHand done!")
############################################################################################################
############################################################################################################ office
for num in range(1, 3+1):
    img_list = glob("data/frames/train/office/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/office/obj_left{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/office/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/office/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("office LHand done!")

for num in range(1, 3+1):
    img_list = glob("data/frames/train/office/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/office/obj_right{}.npy".format(num))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/train/office/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/train/office/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            train_dataset_list.append([[im1, im2], label])
            train_X_list.append([im1, im2])
            train_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("office RHand done!")
############################################################################################################
print(len(train_dataset_list), len(train_X_list), len(train_y_list))
#*****************************************************************************************************************

with open("save/datasets/train_X_list.pickle", "wb") as f:
    pickle.dump(train_X_list, f)
with open("save/datasets/train_y_list.pickle", "wb") as f:
    pickle.dump(train_y_list, f)

