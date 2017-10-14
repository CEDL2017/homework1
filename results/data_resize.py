# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 15:32:52 2017

@author: nvlab
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 12:13:40 2017

@author: nvlab
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:58:21 2017

@author: nvlab
"""

import numpy as np
from PIL import Image
import scipy.misc

global num_house1_L,num_house2_L,num_house3_L,num_lab1_L,num_lab2_L,num_lab3_L,num_lab4_L
global num_house1_R,num_house2_R,num_house3_R,num_lab1_R,num_lab2_R,num_lab3_R,num_lab4_R
global num_off1_R, num_off2_R, num_off3_R, num_off1_L, num_off2_L, num_off3_L
global img_path
num_house1_L = 1
num_off1_L = 1
num_off1_R = 1
num_house2_L = 1
num_off2_L = 1
num_off2_R = 1
num_house3_L = 1
num_off3_L = 1
num_off3_R = 1
num_lab1_L = 1
num_lab2_L = 1
num_lab3_L = 1
num_lab4_L = 1
num_house1_R = 1
num_house2_R = 1
num_house3_R = 1
num_lab1_R = 1
num_lab2_R = 1
num_lab3_R = 1
num_lab4_R = 1
def read_data(batch):

    global num_house1_L,num_house2_L,num_house3_L,num_lab1_L,num_lab2_L,num_lab3_L,num_lab4_L
    global num_house1_R,num_house2_R,num_house3_R,num_lab1_R,num_lab2_R,num_lab3_R,num_lab4_R
    global num_off1_R, num_off2_R, num_off3_R, num_off1_L, num_off2_L, num_off3_L
    global img_path
    for i in range(batch):
        if num_house1_L<=831:        
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/1/Lhand/Image%s.png"%(num_house1_L)
            num_house1_L+=1
        elif num_house1_R<=831:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/1/Rhand/Image%s.png"%(num_house1_R)
            num_house1_R+=1
        elif num_house2_L<=988:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/2/Lhand/Image%s.png"%(num_house2_L)
            num_house2_L+=1
        elif num_house2_R<=988:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/2/Rhand/Image%s.png"%(num_house2_R)
            num_house2_R+=1
        elif num_house3_L<=1229:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/3/Lhand/Image%s.png"%(num_house3_L)
            num_house3_L+=1
        elif num_house3_R<=1229:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/house/3/Rhand/Image%s.png"%(num_house3_R)
            num_house3_R+=1
        elif num_lab1_L<=501:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/1/Lhand/Image%s.png"%(num_lab1_L)
            num_lab1_L+=1
        elif num_lab1_R<=501:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/1/Rhand/Image%s.png"%(num_lab1_R)
            num_lab1_R+=1
        elif num_lab2_L<=589:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/2/Lhand/Image%s.png"%(num_lab2_L)
            num_lab2_L+=1
        elif num_lab2_R<=589:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/2/Rhand/Image%s.png"%(num_lab2_R)
            num_lab2_R+=1
        elif num_lab3_L<=730:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/3/Lhand/Image%s.png"%(num_lab3_L)
            num_lab3_L+=1
        elif num_lab3_R<=730:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/3/Rhand/Image%s.png"%(num_lab3_R)
            num_lab3_R+=1
        elif num_lab4_L<=660:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/4/Lhand/Image%s.png"%(num_lab4_L)
            num_lab4_L+=1
        elif num_lab4_R<=660:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/lab/4/Rhand/Image%s.png"%(num_lab4_R)
            num_lab4_R+=1
        elif num_off1_L<=745:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/1/Lhand/Image%s.png"%(num_off1_L)
            num_off1_L+=1
        elif num_off1_R<=745:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/1/Rhand/Image%s.png"%(num_off1_R)
            num_off1_R+=1
        elif num_off2_L<=572:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/2/Lhand/Image%s.png"%(num_off2_L)
            num_off2_L+=1
        elif num_off2_R<=572:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/2/Rhand/Image%s.png"%(num_off2_R)
            num_off2_R+=1
        elif num_off3_L<=651:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/3/Lhand/Image%s.png"%(num_off3_L)
            num_off3_L+=1
        elif num_off3_R<=651:
            img_path = "../../../../Disk2/cedl/handcam/frames/train/office/3/Rhand/Image%s.png"%(num_off3_R)
            num_off3_R+=1
        img = Image.open(img_path)
        img = np.array(img)
        x_img = np.resize(img, [128,248,3])
        x_img = x_img/255.0
        
        scipy.misc.imsave('frames_resize/Image%s.png'%i, x_img)
    
def read_data_test(batch):

    global num_house1_L,num_house2_L,num_house3_L,num_lab1_L,num_lab2_L,num_lab3_L,num_lab4_L
    global num_house1_R,num_house2_R,num_house3_R,num_lab1_R,num_lab2_R,num_lab3_R,num_lab4_R
    global num_off1_R, num_off2_R, num_off3_R, num_off1_L, num_off2_L, num_off3_L
    global img_path
    for i in range(batch):
        if num_house1_L<=830:        
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/1/Lhand/Image%s.png"%(num_house1_L)
            num_house1_L+=1
        elif num_house1_R<=830:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/1/Rhand/Image%s.png"%(num_house1_R)
            num_house1_R+=1
        elif num_house2_L<=887:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/2/Lhand/Image%s.png"%(num_house2_L)
            num_house2_L+=1
        elif num_house2_R<=887:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/2/Rhand/Image%s.png"%(num_house2_R)
            num_house2_R+=1
        elif num_house3_L<=929:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/3/Lhand/Image%s.png"%(num_house3_L)
            num_house3_L+=1
        elif num_house3_R<=929:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/house/3/Rhand/Image%s.png"%(num_house3_R)
            num_house3_R+=1
        elif num_lab1_L<=539:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/1/Lhand/Image%s.png"%(num_lab1_L)
            num_lab1_L+=1
        elif num_lab1_R<=539:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/1/Rhand/Image%s.png"%(num_lab1_R)
            num_lab1_R+=1
        elif num_lab2_L<=658:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/2/Lhand/Image%s.png"%(num_lab2_L)
            num_lab2_L+=1
        elif num_lab2_R<=658:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/2/Rhand/Image%s.png"%(num_lab2_R)
            num_lab2_R+=1
        elif num_lab3_L<=467:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/3/Lhand/Image%s.png"%(num_lab3_L)
            num_lab3_L+=1
        elif num_lab3_R<=467:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/3/Rhand/Image%s.png"%(num_lab3_R)
            num_lab3_R+=1
        elif num_lab4_L<=503:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/4/Lhand/Image%s.png"%(num_lab4_L)
            num_lab4_L+=1
        elif num_lab4_R<=503:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/lab/4/Rhand/Image%s.png"%(num_lab4_R)
            num_lab4_R+=1
        elif num_off1_L<=590:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/1/Lhand/Image%s.png"%(num_off1_L)
            num_off1_L+=1
        elif num_off1_R<=590:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/1/Rhand/Image%s.png"%(num_off1_R)
            num_off1_R+=1
        elif num_off2_L<=419:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/2/Lhand/Image%s.png"%(num_off2_L)
            num_off2_L+=1
        elif num_off2_R<=419:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/2/Rhand/Image%s.png"%(num_off2_R)
            num_off2_R+=1
        elif num_off3_L<=566:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/3/Lhand/Image%s.png"%(num_off3_L)
            num_off3_L+=1
        elif num_off3_R<=566:
            img_path = "../../../../Disk2/cedl/handcam/frames/test/office/3/Rhand/Image%s.png"%(num_off3_R)
            num_off3_R+=1
        img = Image.open(img_path)
        img = np.array(img)
        x_img = np.resize(img, [128,248,3])
        x_img = x_img/255.0
        
        scipy.misc.imsave('test_resize/Image%s.png'%i, x_img)
    
read_data_test(12776)
read_data(14992)