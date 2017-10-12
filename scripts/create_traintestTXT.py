import os 
import numpy as np
import pandas as pd
os.chdir('D:/Lab/CEDL/hw1/scripts/')
main_dir = '../data/'
envs = ['house','lab','office']

label_name = 'labels/'
label_device = ['obj_left','obj_right']
# label_device = ['obj_left']
TWOSTRING = True

data_name = 'test/'
data_part = ['1','2','3','4']
data_device = ['Lhand','Rhand']
# data_device = ['head']

## note for the pre-process of head-data:
'''
for head 'image' and 'label', I arbitrary select left-hand label as its correponding head-label, since I will 
not use the label of the head in the parallel structure, so it doesn;t matter.
'''

#%%
total_val_num = 0
total_train_num = 0
for _, env in enumerate(envs):
    for idx, device in enumerate(label_device):
        for _, part in enumerate(data_part):
            if env != 'lab' and part == '4':
                continue 
            if data_name=='test/' and env =='lab':
                label_f_dir = main_dir+label_name+env+'/'+device+str(int(part)+4)+'.npy' # label test
            elif data_name=='test/':
                label_f_dir = main_dir+label_name+env+'/'+device+str(int(part)+3)+'.npy' # label test
            else:
                label_f_dir = main_dir+label_name+env+'/'+device+part+'.npy'    # label train
            label_array = np.load(label_f_dir)
            print('now reading %s' % label_f_dir)
            # img_num = len(label_array)
            
            for i, label in enumerate(label_array):
                train_num = int(len(label_array)*0.7)
                val_num = len(label_array) - train_num    
            
                '''
                if i < train_num:
                    with open("head_train.txt", "a") as text_file:
                        f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
                        cores_label = str(int(label))
                        text_file.write(f_dir+' '+cores_label+'\n')
                        total_train_num += 1
                else:
                    with open("head_val.txt", "a") as text_file:
                        f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
                        cores_label = str(int(label))
                        text_file.write(f_dir+' '+cores_label+'\n')
                        total_val_num += 1
                '''
                with open("hand_head_test.txt", "a") as text_file:
                    coresLabel = str(int(label))
                    handDir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
                    if TWOSTRING:
                        headDir = main_dir+data_name+env+'/'+part+'/'+'head'+'/'+'Image'+str(i+1)+'.png'
                        line = ' '.join([handDir, headDir, coresLabel])
                    else:
                        line = ' '.join([handDir, coresLabel])
                    text_file.write(line+'\n')
                    total_train_num += 1
    
    
# print('total_val_num = ',total_val_num)
# print('total_train_num = ',total_train_num)

## Shuffle the txt-file
# data = pd.read_csv('output_list.txt', sep=" ", header=None)
# data.columns = ["a", "b", "c", "etc."]