import glob
import os
import numpy as np
import re
from PIL import Image
from sklearn.model_selection import train_test_split


        
def read_train_data_list(num_classes):
    train_path = os.path.join('frames','train')
    head_list = []
    Lhand_list = []
    Rhand_list = []
    Lhand_labels_list = []
    Rhand_labels_list = []
    places = {'house':3, 'lab':4, 'office':3}
    place_meta_list = []
    for i, place in enumerate(places.keys()):
        for j in range (1, places[place]+1):
            tmp_list = glob.glob(os.path.join(train_path,'head', place, str(j), '*.png'))
            head_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            
            tmp_list = glob.glob(os.path.join(train_path, 'Lhand', place, str(j), '*.png'))
            Lhand_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            
            tmp_list = glob.glob(os.path.join(train_path, 'Rhand', place, str(j), '*.png'))
            Rhand_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            
            Lhand_labels_list.append(np.load(os.path.join('labels', place, 'obj_left'+str(j)+'.npy')))
            Rhand_labels_list.append(np.load(os.path.join('labels', place, 'obj_right'+str(j)+'.npy')))
            place_meta_list.append(np.full_like(tmp_list, i, np.int8))
    
    
    head_list = head_list + head_list
    hand_list = Lhand_list + Rhand_list
    
    X = np.array(hand_list)
    h = np.array(head_list)
    
    place_meta_list += place_meta_list
    place_meta = np.concatenate(place_meta_list)
    place_meta = np.eye(3)[place_meta]
    hand_meta = np.concatenate((np.full_like(Lhand_list, 0, np.int8),
                               np.full_like(Rhand_list, 1, np.int8)))
    hand_meta = np.eye(2)[hand_meta]
    meta = np.concatenate((place_meta, hand_meta), axis = 1)
    
    Lhand_labels = np.concatenate((Lhand_labels_list))
    Rhand_labels = np.concatenate((Rhand_labels_list))
    
    y = np.concatenate((Lhand_labels, Rhand_labels)).astype(np.int16)
    y = np.eye(num_classes)[y]
    
    X_train, X_test, y_train, y_test, h_train, h_test, meta_train, meta_test = train_test_split(X, y, h, meta, test_size = 0.2)

    return X_train, y_train, h_train, meta_train, X_test, y_test, h_test, meta_test

def read_batch_data(hand_list):
    im_hand = np.array([np.array(Image.open(_)) for _ in hand_list], dtype = np.float32)
    return im_hand

def read_test_data_list(num_classes):
    test_path = os.path.join('frames','test')
    head_list = []
    Lhand_list = []
    Rhand_list = []
    Lhand_labels_list = []
    Rhand_labels_list = []
    places = {'house':3, 'lab':4, 'office':3}
    place_meta_list = []
    
    for i, place in enumerate(places.keys()):
        for j in range (1, places[place]+1):
            tmp_list = glob.glob(os.path.join(test_path,'head', place, str(j), '*.png'))
            head_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            
            tmp_list = glob.glob(os.path.join(test_path, 'Lhand', place, str(j), '*.png'))
            Lhand_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            tmp_list = glob.glob(os.path.join(test_path, 'Rhand', place, str(j), '*.png'))
            Rhand_list += sorted(tmp_list, key=lambda x: int(re.sub('\D', '', x)))
            
            Lhand_labels_list.append(np.load(os.path.join('labels', place, 'obj_left'+str(j+places[place])+'.npy')))
            Rhand_labels_list.append(np.load(os.path.join('labels', place, 'obj_right'+str(j+places[place])+'.npy')))
            place_meta_list.append(np.full_like(tmp_list, i, np.int8))
           
    
    
    head_list = head_list + head_list
    hand_list = Lhand_list + Rhand_list

    x = np.array(hand_list)
    h = np.array(head_list)
    
    place_meta_list += place_meta_list
    place_meta = np.concatenate(place_meta_list)
    place_meta = np.eye(3)[place_meta]
    hand_meta = np.concatenate((np.full_like(Lhand_list, 0, np.int8),
                               np.full_like(Rhand_list, 1, np.int8)))
    hand_meta = np.eye(2)[hand_meta]
    meta = np.concatenate((place_meta, hand_meta), axis = 1)
    
    Lhand_labels = np.concatenate((Lhand_labels_list))
    Rhand_labels = np.concatenate((Rhand_labels_list))
    y = np.concatenate((Lhand_labels, Rhand_labels)).astype(np.uint)

    return x, y, h, meta
