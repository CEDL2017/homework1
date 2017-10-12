import re
import os
import sys
import pandas as pd
import numpy as np


def gen_index(setting_index):
    train_index=[]
    test_index =[]
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

def load_image_paths(image_folder_path='dataset/frames/',
                     data_type='train',
                     label_type='obj',
                     image_type = 'left'):
    
    # Re-format to pre-defined folder names.
    if image_type == 'left':
        image_type = 'Lhand'
    elif image_type == 'right':
        image_type = 'Rhand'
    else:
        image_type = 'head'
    
    image_paths = []
    
    # Hard-coded due to the given dataset splitted by the original author.
    if data_type == 'train':
        index, _ = gen_index(setting_index=0)
    elif data_type == 'test':
        _, index = gen_index(setting_index=0)
        
        # The given `split_id` of image and label in test data are not consistant.
        # e.g. "dataset/frames/test/lab/1" <=> "dataset/labels/lab/xxx_xxx*5*.npy"
        # Based on this example, the generated test `split_id` will started from *5*,
        # we need to let `split_id` start from *1*.
        split_num = {'lab': 0, 'office': 0, 'house': 0}
        
        for tup in index:
            scene_type = tup[0]
            split_num[scene_type] += 1
            
        for i, tup in enumerate(index):
            scene_type, split_id = tup
            tmp_split_id = split_id % split_num[scene_type] 
            split_id = tmp_split_id if tmp_split_id > 0 else split_num[scene_type]
            index[i] = (scene_type, split_id) 

    # Load image paths.
    for tup in index:
        scene_type, split_id = tup
        target_folder_path = os.path.join(image_folder_path, data_type, scene_type, str(split_id), image_type)
        file_names = os.listdir(target_folder_path)
        file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
        image_paths.extend([os.path.join(target_folder_path, file_name) for file_name in file_names])
        
        
    return image_paths
        
def load_labels(label_folder_path='dataset/labels',
                data_type='train',
                label_type='obj',
                hand_type = 'left'):
    """ Note that we do not have labels for head images. """
    
    labels = []
    
    # Hard-coded due to the given dataset splitted by the original author.
    if data_type == 'train':
        index, _ = gen_index(setting_index=0)
    elif data_type == 'test':
        _, index = gen_index(setting_index=0)

    # Load labels
    for tup in index:
        scene_type, split_id = tup
        label_npy_path = os.path.join(label_folder_path, scene_type, 
                                      '{}_{}{}.npy'.format(label_type, hand_type, split_id))
        
        label_npy = np.load(label_npy_path)
        labels.append(label_npy)

    labels = np.concatenate(labels)
    
    return labels

def load_examples(image_folder_path='dataset/frames/',
                  label_folder_path='dataset/labels',
                  data_type='train',
                  label_type='obj',
                  hand_type='left',
                  with_head=False):
    
    if hand_type=='both':
        l_paths = load_image_paths(image_folder_path=image_folder_path, 
                                            data_type=data_type, 
                                            label_type=label_type,
                                            image_type='left')
        r_paths = load_image_paths(image_folder_path=image_folder_path, 
                                            data_type=data_type, 
                                            label_type=label_type,
                                            image_type='right')

        l_labels = load_labels(label_folder_path=label_folder_path,
                         data_type=data_type,
                         label_type=label_type,
                         hand_type='left')

        r_labels = load_labels(label_folder_path=label_folder_path,
                         data_type=data_type,
                         label_type=label_type,
                         hand_type='right')



    else:
        hand_image_paths = load_image_paths(image_folder_path=image_folder_path, 
                                            data_type=data_type, 
                                            label_type=label_type,
                                            image_type=hand_type)
        labels = load_labels(label_folder_path=label_folder_path,
                         data_type=data_type,
                         label_type=label_type,
                         hand_type=hand_type)


    
    if with_head:
        head_image_paths = load_image_paths(image_folder_path=image_folder_path, 
                                        data_type=data_type, 
                                        label_type=label_type,
                                        image_type='head')
    
    
    if hand_type=='both':
        if with_head:
            return head_image_paths, l_paths, r_paths, l_labels, r_labels
        else:
            return l_paths, r_paths, l_labels, r_labels
    else:
        if with_head:
            return head_image_paths, hand_image_paths, labels
        else:
            return hand_image_paths, labels
