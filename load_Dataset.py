import numpy as np

import pickle
import os
import collections
import random
import matplotlib.pyplot as plt
import pdb
import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, utils                              
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

class HandCamDataset(Dataset):

    def __init__(self, stage):
      
        self.image_path = 'data/frames/'+stage+'/'
        self.label_path = 'data/labels/'

        self.scene_list = ['house','lab','office']
        self.video_list = ['1','2','3','4']
        self.handview_list = ['Lhand','Rhand']

        self.image_list = []

	for scene in self.scene_list:
            for video in self.video_list:
                if video == '4' and scene != 'lab':
                    break
                for handview in self.handview_list:
                    print('scene', scene, 'video', video, 'view', handview)
                    frame_list = glob.glob(self.image_path +scene+'/'+video+'/'+handview+'/*')
                    frame_list.sort(key=lambda x:int(x.split('Image')[1].split('.')[0]))
                    self.image_list.extend(frame_list)
        self.labels = np.array([])




        for scene in self.scene_list:
            for video in self.video_list:
                for handview in self.handview_list:
                    if video == '4' and scene != 'lab':
                        break
                    label_index = video
                    if stage == 'test':
                        if scene == 'lab':
                            lable_index = str(int(label_index) + 4)
                        else:
                            lable_index = str(int(label_index) + 3)
                    handview = 'left' if handview == 'Lhand' else 'right'

                    print self.label_path + scene + '/obj_' + handview + label_index + '.npy'
                    self.labels = np.append(self.labels, np.load(self.label_path + scene + '/obj_' + handview + label_index + '.npy'))        
	
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],             
                                        std=[0.229, 0.224, 0.225])
        if stage == 'train':

            self.transform = transforms.Compose([
                transforms.Scale(224),
                #transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        elif stage == 'test':
            self.transform = transforms.Compose([
                transforms.Scale(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.image_list[index]).convert('RGB'))
        label = torch.LongTensor([int(self.labels[index])])

        return image, label

def image_data_loader(args):
	kwargs = {'num_workers': args.workers, 'pin_memory': True}
	train_dataset = HandCamDataset('train')
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, **kwargs)
	
	val_dataset = HandCamDataset('test')
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, **kwargs)

	return train_loader, val_loader

if __name__ == '__main__':
	train_data = HandCamDataset('train')
	test_data = HandCamDataset('test') 

