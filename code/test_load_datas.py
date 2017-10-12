import numpy as np
from load_datas import *
import pdb
import os

label_path='../labels'

# load all labels
train_labels0, test_labels0 = load_all_labels(label_path, 'obj',0)
#train_labels0:12array test_labels0:8array

# load train/test labels automatically
train_labels = load_train_labels(label_path,'obj',setting_index=0)
test_labels = load_test_labels(label_path,'obj',setting_index=0)

# load particular index
# You can load only part of the directories, the index of directories should be a list.
# Such like [(location1, id), (location2, id) ...], ex. [('lab',1), ('lab',2), ('office',1)]
# No setting_index required now.

#train_labels = load_train_labels(label_path,'FA',index=[['lab',1],['lab',2],['office',3]])
#test_labels = load_test_labels(label_path,'FA',index=[['lab',1],['lab',2],['office',3]])

#print(train_labels0)
#print(len(train_labels))
print(type(train_labels[0]))

for i in range(len(train_labels)):
    if np.count_nonzero(train_labels0[i] == train_labels[i]) != len(train_labels0[i]):
        print ('train_labels : i=%d error' % i)

for i in range(len(test_labels)):
    if np.count_nonzero(test_labels0[i] == test_labels[i]) != len(test_labels[i]):
        print ('test_labels: i=%d error' % i)


