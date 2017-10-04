import re
import os
import sys
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
    
    hand_image_paths = load_image_paths(image_folder_path=image_folder_path, 
                                        data_type=data_type, 
                                        label_type=label_type,
                                        image_type=hand_type)
    
    if with_head:
        head_image_paths = load_image_paths(image_folder_path=image_folder_path, 
                                        data_type=data_type, 
                                        label_type=label_type,
                                        image_type='head')
    
    labels = load_labels(label_folder_path=label_folder_path,
                         data_type=data_type,
                         label_type=label_type,
                         hand_type=hand_type)
    
    labels = labels.astype(np.int32)
    
    if with_head:
        return head_image_paths, hand_image_paths, labels
    else:
        return hand_image_paths, labels
    
def my_train_test_split(image_paths, labels, test_size=0.2):
    num_lab = 0
    num_office = 0
    num_house = 0
        
    for image_path in image_paths:
        
        f_names = image_path.split('/')
        frame_folder_idx = [i for i, name in enumerate(f_names) if name == 'frames'][0]
        scene = f_names[frame_folder_idx+2]
        
        if scene == 'lab': num_lab += 1
        elif scene == 'office': num_office += 1
        elif scene == 'house': num_house += 1
            
    lab_image_paths = image_paths[:num_lab]
    lab_labels = labels[:num_lab]
    
    office_image_paths = image_paths[num_lab:num_lab+num_office]
    office_labels = labels[num_lab:num_lab+num_office]
    
    house_image_paths = image_paths[num_lab+num_office:num_lab+num_office+num_house]
    house_labels = labels[num_lab+num_office:num_lab+num_office+num_house] 
    
    lab_train_size = round((1 - test_size)*num_lab)
    office_train_size = round((1 - test_size)*num_office)
    house_train_size = round((1 - test_size)*num_house)
    
    train_image_paths = lab_image_paths[:lab_train_size] + office_image_paths[:office_train_size] + house_image_paths[:house_train_size]
    train_labels = np.concatenate([lab_labels[:lab_train_size],
                                  office_labels[:office_train_size],
                                  house_labels[:house_train_size]])
    
    test_image_paths = lab_image_paths[lab_train_size:] + office_image_paths[office_train_size:] + house_image_paths[house_train_size:]
    test_labels = np.concatenate([lab_labels[lab_train_size:],
                                  office_labels[office_train_size:],
                                  house_labels[house_train_size:]])
    
    return train_image_paths, test_image_paths, train_labels, test_labels

def load_dataset(image_folder_path='dataset/frames/',
                 label_folder_path='dataset/labels/',
                 label_type='obj',
                 hand_types=['left', 'right'],
                 validation_split_ratio=0.2):
    
    train_image_paths, train_labels = [], np.array([])
    val_image_paths, val_labels = [], np.array([])
    test_image_paths, test_labels = load_examples(image_folder_path=image_folder_path,
                                                  label_folder_path=label_folder_path,
                                                  data_type='test', label_type=label_type)
    
    if 'left' in hand_types:
        l_image_paths, l_labels = load_examples(image_folder_path=image_folder_path,
                                                label_folder_path=label_folder_path,
                                                data_type='train', label_type=label_type,
                                                hand_type='left')

        l_train_image_paths, l_val_image_paths, l_train_labels, l_val_labels = \
            my_train_test_split(l_image_paths, l_labels, test_size=validation_split_ratio)
            
        train_image_paths += l_train_image_paths
        train_labels = np.concatenate([train_labels, l_train_labels])
        val_image_paths += l_val_image_paths
        val_labels = np.concatenate([val_labels, l_val_labels])
    
    if 'right' in hand_types:
        r_image_paths, r_labels = load_examples(image_folder_path=image_folder_path,
                                                label_folder_path=label_folder_path,
                                                data_type='train', label_type=label_type,
                                                hand_type='right')

        r_train_image_paths, r_val_image_paths, r_train_labels, r_val_labels = \
            my_train_test_split(r_image_paths, r_labels, test_size=validation_split_ratio)
            
           
        train_image_paths += r_train_image_paths
        train_labels = np.concatenate([train_labels, r_train_labels])
        val_image_paths += r_val_image_paths
        val_labels = np.concatenate([val_labels, r_val_labels])
    
    print('[Train] data/label: {}/{}'.format(len(train_image_paths), len(train_image_paths)))
    print('[Validation] data/label: {}/{}'.format(len(val_image_paths), len(val_labels)))
    print('[Test] data/label: {}/{}'.format(len(test_image_paths), len(test_labels)))
    
    train_labels = train_labels.astype(np.int32)
    val_labels = val_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    
    return train_image_paths, train_labels, val_image_paths, val_labels, test_image_paths, test_labels


        
        
        
    
    
    
 