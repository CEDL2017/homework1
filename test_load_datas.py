import numpy as np
from load_datas import *
import pdb
import re
path_data = os.environ.get("GRAPE_DATASET_DIR")

#label_path='/home/yoooosing/CEDL/hw1/labels/'
label_path = os.path.join(path_data,'labels')


# load all labels
train_labels0, test_labels0 = load_all_labels(label_path, 'obj',0)
#print len(train_labels0)
#print len(test_labels0)
# load train/test labels automatically
#train_labels = load_train_labels(label_path,'obj',setting_index=1)
#test_labels = load_test_labels(label_path,'obj',setting_index=1)

# load particular index
# You can load only part of the directories, the index of directories should be a list.
# Such like [(location1, id), (location2, id) ...], ex. [('lab',1), ('lab',2), ('office',1)]
# No setting_index required now.
# train_labels = load_train_labels(label_path,'FA',index=[['lab',1],['lab',2],['office',3]])
# test_labels = load_test_labels(label_path,'FA',index=[['lab',1],['lab',2],['office',3]])



# load train/test images

train_images = []
duplicate_train_labels = []
#image_path1='/home/yoooosing/CEDL/hw1/train/lab'
#image_path2='/home/yoooosing/CEDL/hw1/train/office'
#image_path3='/home/yoooosing/CEDL/hw1/train/house'

image_path1 = os.path.join(path_data, 'frames/train/lab')
image_path2 = os.path.join(path_data, 'frames/train/office')
image_path3 = os.path.join(path_data, 'frames/train/house')



for i in xrange(1,5):
    temp_path_l = os.listdir(os.path.join(image_path1,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path1,str(i),'Lhand',t) for t in temp_path_l]
    train_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path1,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path1,str(i),'Rhand',t) for t in temp_path_r]
    train_images.extend(temp_path_r)
    
for i in xrange(1,4):
    temp_path_l = os.listdir(os.path.join(image_path2,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path2,str(i),'Lhand',t) for t in temp_path_l]
    train_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path2,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path2,str(i),'Rhand',t) for t in temp_path_r]
    train_images.extend(temp_path_r)

for i in xrange(1,4):
    temp_path_l = os.listdir(os.path.join(image_path3,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path3,str(i),'Lhand',t) for t in temp_path_l]
    train_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path3,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path3,str(i),'Rhand',t) for t in temp_path_r]
    train_images.extend(temp_path_r)

for i in xrange(0,8):
    duplicate_train_labels.extend(train_labels0[i])
for i in xrange(8,14):
    duplicate_train_labels.extend(train_labels0[i])
for i in xrange(14,20):
    duplicate_train_labels.extend(train_labels0[i])

test_images = []
duplicate_test_labels = []
#image_path1='/home/yoooosing/CEDL/hw1/test/lab'
#image_path2='/home/yoooosing/CEDL/hw1/test/office'
#image_path3='/home/yoooosing/CEDL/hw1/test/house'

image_path1 = os.path.join(path_data, 'frames/test/lab')
image_path2 = os.path.join(path_data, 'frames/test/office')
image_path3 = os.path.join(path_data, 'frames/test/house')

for i in xrange(1,5):
    temp_path_l = os.listdir(os.path.join(image_path1,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path1,str(i),'Lhand',t) for t in temp_path_l]
    test_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path1,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path1,str(i),'Rhand',t) for t in temp_path_r]
    test_images.extend(temp_path_r)
    
for i in xrange(1,4):
    temp_path_l = os.listdir(os.path.join(image_path2,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path2,str(i),'Lhand',t) for t in temp_path_l]
    test_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path2,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path2,str(i),'Rhand',t) for t in temp_path_r]
    test_images.extend(temp_path_r)

for i in xrange(1,4):
    temp_path_l = os.listdir(os.path.join(image_path3,str(i),'Lhand'))
    temp_path_l = sorted(temp_path_l, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_l = [os.path.join(image_path3,str(i),'Lhand',t) for t in temp_path_l]
    test_images.extend(temp_path_l)
    temp_path_r = os.listdir(os.path.join(image_path3,str(i),'Rhand'))
    temp_path_r = sorted(temp_path_r, key=lambda x: int(re.sub('\D', '', x)))
    temp_path_r = [os.path.join(image_path3,str(i),'Rhand',t) for t in temp_path_r]
    test_images.extend(temp_path_r)

for i in xrange(0,8):
    duplicate_test_labels.extend(test_labels0[i])
for i in xrange(8,14):
    duplicate_test_labels.extend(test_labels0[i])
for i in xrange(14,20):
    duplicate_test_labels.extend(test_labels0[i])

#print '\n'.join(train_images)
#print len(train_images)
#print len(duplicate_train_labels)
#print len(test_images)
#print len(duplicate_test_labels)


#for i in xrange(len(train_labels)):
#    if np.count_nonzero(train_labels0[i] == train_labels[i]) != len(train_labels0[i]):
#        print 'error'

#for i in xrange(len(test_labels)):
#    if np.count_nonzero(test_labels0[i] == test_labels[i]) != len(test_labels[i]):
#        print 'error'
