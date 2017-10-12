from config import *
from coder import *
import numpy as np
from keras.utils import np_utils
import random
import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.interpolation import zoom
from skimage import transform
def compare(a,b):
    # print ([a,b])
    if len(a)>len(b):
        return 1
    elif len(a)==len(b):
        rslt = sorted([a,b])
        if(rslt[1]==a):
            return 1
        else:
            return -1
    else:
        return -1
def readScoreNetData(j):
    """
        Read the training data for the ScoreNet

        Ret:    The training data tuple of the x and y
    """
    # Load name of files
    # img_name_list = sorted(os.listdir(scorenet_img_path))
    # img_name_list = sorted(os.listdir(scorenet_img_path),cmp=compare)
    img_name_list = sorted(os.listdir(scorenet_img_path), key=lambda x: int(re.sub('\D', '', x)))

    if j == 0 or j == 1:
        img_name_list = sorted(os.listdir(scorenet_img_path_house1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house1 = np.append(img_name_list_house1,'Image'+ str(i+1)+'.png')
    # for i in range(0,988):
    elif j == 2 or j == 3:
        img_name_list = sorted(os.listdir(scorenet_img_path_house2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house2 = np.append(img_name_list_house2,'Image'+ str(i+1)+'.png')
    # for i in range(0,1229):
    elif j == 4 or j == 5:
        img_name_list = sorted(os.listdir(scorenet_img_path_house3L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house3 = np.append(img_name_list_house3,'Image'+ str(i+1)+'.png')

    # for i in range(0,501):
    elif j == 6 or j == 7:
        img_name_list = sorted(os.listdir(scorenet_img_path_lab1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab1 = np.append(img_name_list_lab1,'Image'+ str(i+1)+'.png')
    # for i in range(0,589):
    elif j == 8 or j == 9:
        img_name_list = sorted(os.listdir(scorenet_img_path_lab2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab2 = np.append(img_name_list_lab2,'Image'+ str(i+1)+'.png')
    # for i in range(0,730):
    elif j == 10 or j == 11:
        img_name_list = sorted(os.listdir(scorenet_img_path_lab3L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab3 = np.append(img_name_list_lab3,'Image'+ str(i+1)+'.png')
    # for i in range(0,660):
    elif j == 12 or j == 13:
        img_name_list = sorted(os.listdir(scorenet_img_path_lab4L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab4 = np.append(img_name_list_lab4,'Image'+ str(i+1)+'.png')

    # for i in range(0,745):
    elif j == 14 or j == 15:
        img_name_list = sorted(os.listdir(scorenet_img_path_office1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_office1 = np.append(img_name_list_office1,'Image'+ str(i+1)+'.png')
    # for i in range(0,572):
    elif j == 16 or j == 17:
        img_name_list = sorted(os.listdir(scorenet_img_path_office2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_office2 = np.append(img_name_list_office2,'Image'+ str(i+1)+'.png')
    # for i in range(0,651):
    elif j == 18 or j == 19:
        img_name_list = sorted(os.listdir(scorenet_img_path_office3L), key=lambda x: int(re.sub('\D', '', x)))


    # print(len(img_name_list))
    # dat_name_list = sorted(os.listdir(scorenet_dat_path))
    # dat = []
    # dat = np.load('/home/timyang/Downloads/labels/house/obj_left1.npy')
    dat = []
    if j == 0:
        dat = np.load(scorenet_dat_path_house1L)
    elif j == 1:    
        dat = np.load(scorenet_dat_path_house1R)
    elif j == 2:
        dat = np.load(scorenet_dat_path_house2L)
    elif j == 3:    
        dat = np.load(scorenet_dat_path_house2R)
    elif j == 4:
        dat = np.load(scorenet_dat_path_house3L)
    elif j == 5:
        dat = np.load(scorenet_dat_path_house3R)
    elif j == 6:
        dat = np.load(scorenet_dat_path_lab1L)
    elif j == 7:
        dat = np.load(scorenet_dat_path_lab1R)
    elif j == 8:
        dat = np.load(scorenet_dat_path_lab2L)
    elif j == 9:
        dat = np.load(scorenet_dat_path_lab2R)
    elif j == 10:
        dat = np.load(scorenet_dat_path_lab3L)
    elif j == 11:
        dat = np.load(scorenet_dat_path_lab3R)
    elif j == 12:
        dat = np.load(scorenet_dat_path_lab4L)
    elif j == 13:
        dat = np.load(scorenet_dat_path_lab4L)
    elif j == 14:
        dat = np.load(scorenet_dat_path_office1L)
    elif j == 15:
        dat = np.load(scorenet_dat_path_office1R)
    elif j == 16:
        dat = np.load(scorenet_dat_path_office2L)
    elif j == 17:
        dat = np.load(scorenet_dat_path_office2R)
    elif j == 18:
        dat = np.load(scorenet_dat_path_office3L)
    elif j == 19:
        dat = np.load(scorenet_dat_path_office3R)
    # dat_name_list = int(dat.shape[0])
    # print (dat.shape)
    # dat = dat[692:]
    # print (dat)

    # Shuffle
    for i in range(int(len(img_name_list)/2)):
        swap_index_1 = random.randint(0, len(img_name_list)-1)
        swap_index_2 = random.randint(0, len(img_name_list)-1)
        _ = img_name_list[swap_index_1]
        img_name_list[swap_index_1] = img_name_list[swap_index_2]
        img_name_list[swap_index_2] = _
        _ = dat[swap_index_1]
        dat[swap_index_1] = dat[swap_index_2]
        dat[swap_index_2] = _
        # _ = dat_name_list[swap_index_1]
        # dat_name_list[swap_index_1] = dat_name_list[swap_index_2]
        # dat_name_list[swap_index_2] = _
    # if len(img_name_list) != len(dat_name_list):
    #     print("file distribution is wrong...")
    imgnum=0
    free = 100
    for i in range(len(img_name_list)):
        if dat[i]==0:
            imgnum+=1
    # x_train = np.ndarray([len(img_name_list)-imgnum+free, img_height, img_width, img_channel])
    # y_train = np.ndarray([len(img_name_list)-imgnum+free,len(obj_name_2_index) * grid_height_num * grid_width_num  ])
    x_train = np.ndarray([len(img_name_list), img_height, img_width, img_channel])
    y_train = np.ndarray([len(img_name_list),len(obj_name_2_index) * grid_height_num * grid_width_num  ])
    
    
    # Create object
    free_num = 0
    print('imgnum',imgnum)
    cnt = 0
    # dat_reduce = np.ndarray([len(img_name_list)-imgnum+free,1 ])
    dat_reduce = np.ndarray([len(img_name_list),1 ])
    # dat_reduce = []
    for i in range(len(img_name_list)):
        if j == 0:
            img = mpimg.imread(scorenet_img_path_house1L + img_name_list[i])
        elif j == 1:
            img = mpimg.imread(scorenet_img_path_house1R + img_name_list[i])
        elif j == 2:
            img = mpimg.imread(scorenet_img_path_house2L + img_name_list[i])
        elif j == 3:
            img = mpimg.imread(scorenet_img_path_house2R + img_name_list[i])
        elif j == 4:
            img = mpimg.imread(scorenet_img_path_house3L + img_name_list[i])
        elif j == 5:
            img = mpimg.imread(scorenet_img_path_house3R + img_name_list[i])
        elif j == 6:
            img = mpimg.imread(scorenet_img_path_lab1L + img_name_list[i])
        elif j == 7:
            img = mpimg.imread(scorenet_img_path_lab1R + img_name_list[i])
        elif j == 8:
            img = mpimg.imread(scorenet_img_path_lab2L + img_name_list[i])
        elif j == 9:
            img = mpimg.imread(scorenet_img_path_lab2R + img_name_list[i])
        elif j == 10:
            img = mpimg.imread(scorenet_img_path_lab3L + img_name_list[i])
        elif j == 11:
            img = mpimg.imread(scorenet_img_path_lab3R + img_name_list[i])
        elif j == 12:
            img = mpimg.imread(scorenet_img_path_lab4L + img_name_list[i])
        elif j == 13:
            img = mpimg.imread(scorenet_img_path_lab4R + img_name_list[i])
        elif j == 14:
            img = mpimg.imread(scorenet_img_path_office1L + img_name_list[i])
        elif j == 15:
            img = mpimg.imread(scorenet_img_path_office1R + img_name_list[i])
        elif j == 16:
            img = mpimg.imread(scorenet_img_path_office2L + img_name_list[i])
        elif j == 17:
            img = mpimg.imread(scorenet_img_path_office2R + img_name_list[i])
        elif j == 18:
            img = mpimg.imread(scorenet_img_path_office3L + img_name_list[i])
        elif j == 19:
            img = mpimg.imread(scorenet_img_path_office3R + img_name_list[i])
        # print('free_num % 20',free_num % 20)
        # if dat[i]==0 and free_num == 20:
        # choice = random.randint(0,4)
	


        # if dat[i]==0 and free_num != free:        
        #     free_num += 1
        #     dat_reduce[cnt,...] = dat[i]
        #     if choice == 0:
        #         img = img
        #     elif choice == 1:
        #         img = np.rot90(img, choice)
        #     elif choice == 2:
        #         img = np.rot90(img, choice)
        #     elif choice == 3:
        #         img = np.rot90(img, choice)
        #     elif choice == 4:
        #         img = np.fliplr(img)
        #     img = transform.resize(img,(img_width,img_height))
        #     x_train[cnt, ...] = img
        #     cnt+=1
        # elif dat[i]!=0:
        #     dat_reduce[cnt,...] = dat[i]
        #     if choice == 0:
        #         img = img
        #     elif choice == 1:
        #         img = np.rot90(img, choice)
        #     elif choice == 2:
        #         img = np.rot90(img, choice)
        #     elif choice == 3:
        #         img = np.rot90(img, choice)
        #     elif choice == 4:
        #         img = np.fliplr(img)
        #     img = transform.resize(img,(img_width,img_height))
        #     x_train[cnt, ...] = img
        #     cnt+=1
        # else:
        #     free_num+=1
        # print ('cnt',cnt)
        img = transform.resize(img,(img_width,img_height))
        x_train[i, ...] = img

    
        # x_train[i] = x_train[i]

    # img = cv2.imread(scorenet_img_path + img_name_list[0])
    # print(scorenet_img_path + img_name_list[0])
    # img = cv2.resize(img,(480,270))    
    # height, width, channel = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    # x_train = np.ndarray([len(img_name_list), height, width, channel])
        
    # print ('ytrain_shape:\n',y_train.shape)
    # Fill the list
    # for i in range(len(img_name_list)):
    #     img = cv2.imread(scorenet_img_path + img_name_list[i])
    #     img = cv2.resize(img,(480,270)) 
    #     # vector = encodeByFile(img, scorenet_dat_path + dat_name_list[i])
    #     x_train[i, ...] = img / 255.0
        # y_train[i] = vector
    
    y_train = np_utils.to_categorical(dat,scorenet_fc_num)
    # y_train = np_utils.to_categorical(dat_reduce,scorenet_fc_num)
    print ('ytrain_shape:\n',y_train.shape)
    return (x_train, y_train ,imgnum)

def readTestData(j):
    """
        Read the training data for the ScoreNet

        Ret:    The training data tuple of the x and y
    """
    # Load name of files
    # img_name_list = sorted(os.listdir(scorenet_img_path))
    # img_name_list = sorted(os.listdir(scorenet_img_path),cmp=compare)
    img_name_list = sorted(os.listdir(train_img_path), key=lambda x: int(re.sub('\D', '', x)))

    if j == 0 or j == 1:
        img_name_list = sorted(os.listdir(train_img_path_house1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house1 = np.append(img_name_list_house1,'Image'+ str(i+1)+'.png')
    # for i in range(0,988):
    elif j == 2 or j == 3:
        img_name_list = sorted(os.listdir(train_img_path_house2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house2 = np.append(img_name_list_house2,'Image'+ str(i+1)+'.png')
    # for i in range(0,1229):
    elif j == 4 or j == 5:
        img_name_list = sorted(os.listdir(train_img_path_house3L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_house3 = np.append(img_name_list_house3,'Image'+ str(i+1)+'.png')

    # for i in range(0,501):
    elif j == 6 or j == 7:
        img_name_list = sorted(os.listdir(train_img_path_lab1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab1 = np.append(img_name_list_lab1,'Image'+ str(i+1)+'.png')
    # for i in range(0,589):
    elif j == 8 or j == 9:
        img_name_list = sorted(os.listdir(train_img_path_lab2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab2 = np.append(img_name_list_lab2,'Image'+ str(i+1)+'.png')
    # for i in range(0,730):
    elif j == 10 or j == 11:
        img_name_list = sorted(os.listdir(train_img_path_lab3L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab3 = np.append(img_name_list_lab3,'Image'+ str(i+1)+'.png')
    # for i in range(0,660):
    elif j == 12 or j == 13:
        img_name_list = sorted(os.listdir(train_img_path_lab4L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_lab4 = np.append(img_name_list_lab4,'Image'+ str(i+1)+'.png')

    # for i in range(0,745):
    elif j == 14 or j == 15:
        img_name_list = sorted(os.listdir(train_img_path_office1L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_office1 = np.append(img_name_list_office1,'Image'+ str(i+1)+'.png')
    # for i in range(0,572):
    elif j == 16 or j == 17:
        img_name_list = sorted(os.listdir(train_img_path_office2L), key=lambda x: int(re.sub('\D', '', x)))
        # img_name_list_office2 = np.append(img_name_list_office2,'Image'+ str(i+1)+'.png')
    # for i in range(0,651):
    elif j == 18 or j == 19:
        img_name_list = sorted(os.listdir(train_img_path_office3L), key=lambda x: int(re.sub('\D', '', x)))


    # print(len(img_name_list))
    # dat_name_list = sorted(os.listdir(train_dat_path))
    # dat = []
    # dat = np.load('/home/timyang/Downloads/labels/house/obj_left1.npy')
    dat = []
    if j == 0:
        dat = np.load(train_dat_path_house1L)
    elif j == 1:    
        dat = np.load(train_dat_path_house1R)
    elif j == 2:
        dat = np.load(train_dat_path_house2L)
    elif j == 3:    
        dat = np.load(train_dat_path_house2R)
    elif j == 4:
        dat = np.load(train_dat_path_house3L)
    elif j == 5:
        dat = np.load(train_dat_path_house3R)
    elif j == 6:
        dat = np.load(train_dat_path_lab1L)
    elif j == 7:
        dat = np.load(train_dat_path_lab1R)
    elif j == 8:
        dat = np.load(train_dat_path_lab2L)
    elif j == 9:
        dat = np.load(train_dat_path_lab2R)
    elif j == 10:
        dat = np.load(train_dat_path_lab3L)
    elif j == 11:
        dat = np.load(train_dat_path_lab3R)
    elif j == 12:
        dat = np.load(train_dat_path_lab4L)
    elif j == 13:
        dat = np.load(train_dat_path_lab4L)
    elif j == 14:
        dat = np.load(train_dat_path_office1L)
    elif j == 15:
        dat = np.load(train_dat_path_office1R)
    elif j == 16:
        dat = np.load(train_dat_path_office2L)
    elif j == 17:
        dat = np.load(train_dat_path_office2R)
    elif j == 18:
        dat = np.load(train_dat_path_office3L)
    elif j == 19:
        dat = np.load(train_dat_path_office3R)
    # dat_name_list = int(dat.shape[0])
    # print (dat.shape)
    # dat = dat[692:]
    # print (dat)

    # Shuffle
    for i in range(int(len(img_name_list)/2)):
        swap_index_1 = random.randint(0, len(img_name_list)-1)
        swap_index_2 = random.randint(0, len(img_name_list)-1)
        _ = img_name_list[swap_index_1]
        img_name_list[swap_index_1] = img_name_list[swap_index_2]
        img_name_list[swap_index_2] = _
        _ = dat[swap_index_1]
        dat[swap_index_1] = dat[swap_index_2]
        dat[swap_index_2] = _
        # _ = dat_name_list[swap_index_1]
        # dat_name_list[swap_index_1] = dat_name_list[swap_index_2]
        # dat_name_list[swap_index_2] = _
    # if len(img_name_list) != len(dat_name_list):
    #     print("file distribution is wrong...")
    x_train = np.ndarray([len(img_name_list), img_height, img_width, img_channel])
    y_train = np.ndarray([len(img_name_list),len(obj_name_2_index) * grid_height_num * grid_width_num  ])
    
    # Create object

    for i in range(len(img_name_list)):
        if j == 0:
            img = mpimg.imread(train_img_path_house1L + img_name_list[i])
        elif j == 1:
            img = mpimg.imread(train_img_path_house1R + img_name_list[i])
        elif j == 2:
            img = mpimg.imread(train_img_path_house2L + img_name_list[i])
        elif j == 3:
            img = mpimg.imread(train_img_path_house2R + img_name_list[i])
        elif j == 4:
            img = mpimg.imread(train_img_path_house3L + img_name_list[i])
        elif j == 5:
            img = mpimg.imread(train_img_path_house3R + img_name_list[i])
        elif j == 6:
            img = mpimg.imread(train_img_path_lab1L + img_name_list[i])
        elif j == 7:
            img = mpimg.imread(train_img_path_lab1R + img_name_list[i])
        elif j == 8:
            img = mpimg.imread(train_img_path_lab2L + img_name_list[i])
        elif j == 9:
            img = mpimg.imread(train_img_path_lab2R + img_name_list[i])
        elif j == 10:
            img = mpimg.imread(train_img_path_lab3L + img_name_list[i])
        elif j == 11:
            img = mpimg.imread(train_img_path_lab3R + img_name_list[i])
        elif j == 12:
            img = mpimg.imread(train_img_path_lab4L + img_name_list[i])
        elif j == 13:
            img = mpimg.imread(train_img_path_lab4R + img_name_list[i])
        elif j == 14:
            img = mpimg.imread(train_img_path_office1L + img_name_list[i])
        elif j == 15:
            img = mpimg.imread(train_img_path_office1R + img_name_list[i])
        elif j == 16:
            img = mpimg.imread(train_img_path_office2L + img_name_list[i])
        elif j == 17:
            img = mpimg.imread(train_img_path_office2R + img_name_list[i])
        elif j == 18:
            img = mpimg.imread(train_img_path_office3L + img_name_list[i])
        elif j == 19:
            img = mpimg.imread(train_img_path_office3R + img_name_list[i])

        img = transform.resize(img,(img_width,img_height))
        x_train[i, ...] = img
    imgnum=0
    for i in range(len(img_name_list)):
        if dat[i]==0:
            imgnum+=1
        # x_train[i] = x_train[i]

    # img = cv2.imread(train_img_path + img_name_list[0])
    # print(train_img_path + img_name_list[0])
    # img = cv2.resize(img,(480,270))    
    # height, width, channel = np.shape(img)[0], np.shape(img)[1], np.shape(img)[2]
    # x_train = np.ndarray([len(img_name_list), height, width, channel])
        
    print ('ytrain_shape:\n',y_train.shape)
    # Fill the list
    # for i in range(len(img_name_list)):
    #     img = cv2.imread(train_img_path + img_name_list[i])
    #     img = cv2.resize(img,(480,270)) 
    #     # vector = encodeByFile(img, train_dat_path + dat_name_list[i])
    #     x_train[i, ...] = img / 255.0
        # y_train[i] = vector
    
    y_train = np_utils.to_categorical(dat,scorenet_fc_num)
    
    return (x_train, y_train ,imgnum)

def readUNetData(save_path='../img/unet/'):
    """
        Read the whole training data for the UNet

        Ret:    The training data tuple and the testing data tuple
    """
    # Read Training data
    train_x, train_y = readDataSingleFolder(save_path + 'train/')

    # Read Testing data
    test_x, test_y = readDataSingleFolder(save_path + 'test/')

    return (train_x, train_y), (test_x, test_y)

# def readDataSingleFolder(save_path):
#     """
#         Read the training data from the specific folder for the UNet

#         Ret:    The data tuple
#     """
#     img_name_list = sorted(os.listdir(save_path))
#     if len(img_name_list) % 2 == 1:
#         print ("image number error...")
#         exit()
#     x = np.ndarray([int(len(img_name_list) / 2), int(img_height), int(img_width), img_channel])
#     y = np.ndarray([int(len(img_name_list) / 2), int(img_height), int(img_width), 1])
#     for i in range(int(len(img_name_list))):
#         if img_name_list[i][-5] == 'g':
#             y[int(i/2)] = np.expand_dims(cv2.imread(save_path + img_name_list[i], 0), -1)
#         else:
#             x[int(i/2)] = cv2.imread(save_path + img_name_list[i], 1)
#     return x, y
