import os
import re

image_file = ['/train/house/1/Lhand', '/train/house/1/Rhand', 
             '/train/house/2/Lhand', '/train/house/2/Rhand',
             '/train/house/3/Lhand', '/train/house/3/Rhand',
             '/train/lab/1/Lhand', '/train/lab/1/Rhand', 
             '/train/lab/2/Lhand', '/train/lab/2/Rhand',
             '/train/lab/3/Lhand', '/train/lab/3/Rhand',
             '/train/lab/4/Lhand', '/train/lab/4/Rhand',
             '/train/office/1/Lhand', '/train/office/1/Rhand', 
             '/train/office/2/Lhand', '/train/office/2/Rhand',
             '/train/office/3/Lhand', '/train/office/3/Rhand'
             ]

test_file = ['/test/house/1/Lhand', '/test/house/1/Rhand', 
             '/test/house/2/Lhand', '/test/house/2/Rhand',
             '/test/house/3/Lhand', '/test/house/3/Rhand',
             '/test/lab/1/Lhand', '/test/lab/1/Rhand', 
             '/test/lab/2/Lhand', '/test/lab/2/Rhand',
             '/test/lab/3/Lhand', '/test/lab/3/Rhand',
             '/test/lab/4/Lhand', '/test/lab/4/Rhand',
             '/test/office/1/Lhand', '/test/office/1/Rhand', 
             '/test/office/2/Lhand', '/test/office/2/Rhand',
             '/test/office/3/Lhand', '/test/office/3/Rhand']

label_train_file = ['/labels/house/obj_left1.npy', '/labels/house/obj_left2.npy',
                    '/labels/house/obj_left3.npy', '/labels/house/obj_right1.npy',
                    '/labels/house/obj_right2.npy', '/labels/house/obj_right3.npy',
                    '/labels/lab/obj_left1.npy', '/labels/lab/obj_left2.npy',
                    '/labels/lab/obj_left3.npy', '/labels/lab/obj_left4.npy',
                    '/labels/lab/obj_right1.npy', '/labels/lab/obj_right2.npy',
                    '/labels/lab/obj_right3.npy', '/labels/lab/obj_right4.npy',
                    '/labels/office/obj_left1.npy', '/labels/office/obj_left2.npy',
                    '/labels/office/obj_left3.npy', '/labels/office/obj_right1.npy',
                    '/labels/office/obj_right2.npy', '/labels/office/obj_right3.npy' ]

label_test_file = ['/labels/house/obj_left4.npy', '/labels/house/obj_left5.npy',
                    '/labels/house/obj_left6.npy', '/labels/house/obj_right4.npy',
                    '/labels/house/obj_right5.npy', '/labels/house/obj_right6.npy',
                    '/labels/lab/obj_left5.npy', '/labels/lab/obj_left6.npy',
                    '/labels/lab/obj_left7.npy', '/labels/lab/obj_left8.npy',
                    '/labels/lab/obj_right5.npy', '/labels/lab/obj_right6.npy',
                    '/labels/lab/obj_right7.npy', '/labels/lab/obj_right8.npy',
                    '/labels/office/obj_left4.npy', '/labels/office/obj_left5.npy',
                    '/labels/office/obj_left6.npy', '/labels/office/obj_right4.npy',
                    '/labels/office/obj_right5.npy', '/labels/office/obj_right6.npy' ]


txt = open('train_image.txt', 'w')

for file in image_file:
    unsorted_image =[os.path.join(file, f) for f in os.listdir(file)]    
    unsorted_image = sorted(unsorted_image, key = lambda x:int(re.sub('\D','',x)))
    for data in unsorted_image:
        print(data, file = txt)   

txt = open('test_image.txt', 'w')

for file in test_file:
    unsorted_image =[os.path.join(file, f) for f in os.listdir(file)]    
    unsorted_image = sorted(unsorted_image, key = lambda x:int(re.sub('\D','',x)))
    for data in unsorted_image:
        print(data, file = txt)
        
txt = open('train_label.txt', 'w')

for data in label_train_file:
    print(data, file = txt)


txt = open('test_label.txt', 'w')

for data in label_test_file:
    print(data, file = txt)



