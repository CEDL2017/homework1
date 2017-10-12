import os 
import numpy as np

main_dir = './resize_data/frames/'
envs = ['house','lab','office']

label_name = 'labels/'
label_device = ['obj_left','obj_right']
# label_device = ['obj_left','obj_left']

data_name = 'train/'
data_part = ['1','2','3','4']
# data_name = 'test/'
# data_part = ['4','5','6','7','8']
data_device = ['Lhand','Rhand']
# data_device = ['head','head'] #why here need to duplicate is because we have to make the data amount of head equal to Lhand+Rhead

## note for the pre-process of head-data:
'''
for head 'image' and 'label', I arbitrary select left-hand label as its correponding head-label, since I will 
not use the label of the head in the parallel structure, so it doesn;t matter.

'''

for _, env in enumerate(envs):
	for idx, device in enumerate(label_device):
		for _, part in enumerate(data_part):
			label_f_dir = main_dir+label_name+env+'/'+device+part+'.npy'
			# if (env == 'lab' and part == '4') or (env != 'lab' and (part == '7' or part == '8')):
			if env != 'lab' and part == '4':
				continue 
			label_array = np.load(label_f_dir)
			print('now reading %s' % main_dir+label_name+env+'/'+device+part+'.npy'	)
			# img_num = len(label_array)

			for i, label in enumerate(label_array):
				
				with open("hand_head_all_test.txt", "a") as text_file:
					f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
					f_head_dir = main_dir+data_name+env+'/'+part+'/'+'head'+'/'+'Image'+str(i+1)+'.png'
					cores_label = str(int(label))
					text_file.write(f_dir+' '+ f_head_dir +' '+cores_label+'\n')
					total_train_num += 1
				
				with open("hand_all_test.txt", "a") as text_file:
					f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
					f_head_dir = main_dir+data_name+env+'/'+part+'/'+'head'+'/'+'Image'+str(i+1)+'.png'
					cores_label = str(int(label))
					text_file.write(f_dir+' '+cores_label+'\n')
					total_train_num += 1


				# inappropriate way on divide train/val data
				# train_num = int(len(label_array)*0.7)
				# val_num = len(label_array) - train_num
				
				# if i < train_num:
				# 	with open("hand_head_train.txt", "a") as text_file:
				# 		f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
				# 		f_head_dir = main_dir+data_name+env+'/'+part+'/'+'head'+'/'+'Image'+str(i+1)+'.png'
				# 		cores_label = str(int(label))
				# 		text_file.write(f_dir+' '+ f_head_dir +' '+cores_label+'\n')
				# 		total_train_num += 1
				# else:
				# 	with open("hand_head_val.txt", "a") as text_file:
				# 		f_dir = main_dir+data_name+env+'/'+part+'/'+data_device[idx]+'/'+'Image'+str(i+1)+'.png'
				# 		f_head_dir = main_dir+data_name+env+'/'+part+'/'+'head'+'/'+'Image'+str(i+1)+'.png'
				# 		cores_label = str(int(label))
				# 		text_file.write(f_dir+' '+ f_head_dir +' '+cores_label+'\n')
				# 		total_val_num += 1