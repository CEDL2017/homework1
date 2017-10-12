import os
import re
import numpy as np
def read_data():
    labels = np.array([])
    #first loading the train files, then the test files
    for i in range(2):
        for location in ['house', 'lab', 'office']:
            for hand in ['left', 'right']:            
                for j in range(3):
                    file_name = 'FA_' + hand + str(i * 3 + j + 1) + '.npy'
                    print(file_name)
                    try:
                        label_path = os.path.join('data/labels', location, file_name)
                        label_file = np.load(label_path)
                        labels = np.append(labels, label_file)
                    except:
                        print('error when loading: ', label_path)
    print(labels.shape) #25828
    return labels
labels = read_data()

target_folder_path = 'data/processed_data/train'
file_names = os.listdir(target_folder_path)
file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
filename = 'train.txt'
cnt = 0 
with open(filename, 'w') as out:
    for i in file_names:
        out.write(os.path.join(target_folder_path, i) + ' ' + str(int(labels[int(cnt / 2)])) + '\n')
        cnt += 1
    

target_folder_path = 'data/processed_data/test'
file_names = os.listdir(target_folder_path)
file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
filename = 'val.txt'
with open(filename, 'w') as out:
    for i in file_names:
        out.write(os.path.join(target_folder_path, i) + ' ' + str(int(labels[int(cnt / 2)])) + '\n')
        cnt += 1
