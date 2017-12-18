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
test_dataset_list = []
test_X_list = []
test_y_list = []
#*****************************************************************************************************************
############################################################################################################ house
for num in range(1, 3+1):
    img_list = glob("data/frames/test/house/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/house/obj_left{}.npy".format(num + 3))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/house/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/house/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("house LHand done!")

for num in range(1, 3+1):
    img_list = glob("data/frames/test/house/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/house/obj_right{}.npy".format(num + 3))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/house/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/house/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("house RHand done!")
############################################################################################################
############################################################################################################ lab
for num in range(1, 4+1):
    img_list = glob("data/frames/test/lab/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/lab/obj_left{}.npy".format(num+4))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/lab/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/lab/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("Lab LHand done!")

for num in range(1, 4+1):
    img_list = glob("data/frames/test/lab/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/lab/obj_right{}.npy".format(num + 4))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/lab/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/lab/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("lab RHand done!")
############################################################################################################
############################################################################################################ office
for num in range(1, 3+1):
    img_list = glob("data/frames/test/office/{}/Lhand/*.png".format(num))
    label_npy = np.load("data/labels/office/obj_left{}.npy".format(num+3))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/office/{}/Lhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/office/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("office LHand done!")

for num in range(1, 3+1):
    img_list = glob("data/frames/test/office/{}/Rhand/*.png".format(num))
    label_npy = np.load("data/labels/office/obj_right{}.npy".format(num+3))
    if len(img_list) == len(label_npy):
        print("start append...")
        for i in range(1, len(img_list)+1):
            im1 = "data/frames/test/office/{}/Rhand/Image{}.png".format(num, i)
            im2 = "data/frames/test/office/{}/head/Image{}.png".format(num, i)
            label = int(label_npy[i - 1])
            test_dataset_list.append([[im1, im2], label])
            test_X_list.append([im1, im2])
            test_y_list.append(label)
            # print("done: {}/{}".format(i, len(img_list)))

    else:
        print("warnning!")
    # print(len(dataset_list), len(X_list), len(y_list))
print("office RHand done!")
############################################################################################################
print(len(test_dataset_list), len(test_X_list), len(test_y_list))
#*****************************************************************************************************************

with open("save/datasets/test_X_list.pickle", "wb") as f:
    pickle.dump(test_X_list, f)
with open("save/datasets/test_y_list.pickle", "wb") as f:
    pickle.dump(test_y_list, f)
