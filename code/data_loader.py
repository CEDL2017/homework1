import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import numpy as np
import glob
from PIL import Image
import pdb


class HandCamDataset(data.Dataset):
    def __init__(self, data_dir, split):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.data_dir = data_dir
        self.data = []

        if split == 'train':
            train_label_idx = {'house': ['1', '2', '3'], 'lab': ['1', '2', '3', '4'], 'office': ['1', '2', '3']}

            for place in train_label_idx:
                for idx in train_label_idx[place]:
                    Rframes = glob.glob(os.path.join(data_dir, "frames/train/%s/%s/Rhand/*.png"%(place, idx)))
                    Lframes = glob.glob(os.path.join(data_dir, "frames/train/%s/%s/Lhand/*.png"%(place, idx)))
                    Rframes.sort(key=lambda x:int(x.split('Image')[1].split('.')[0]))
                    Lframes.sort(key=lambda x:int(x.split('Image')[1].split('.')[0]))
                    if len(Rframes) != len(Lframes):
                        pdb.set_trace()
                    Rlabels = np.load(os.path.join(data_dir,'labels/%s/obj_right%s.npy'%(place, idx)))
                    Llabels = np.load(os.path.join(data_dir,'labels/%s/obj_left%s.npy'%(place, idx)))
                    for j in range(len(Rframes)):
                        self.data.append([Rframes[j], Rlabels[j]])
                        self.data.append([Lframes[j], Llabels[j]])
        else:
            test_label_idx = {'house': ['4', '5', '6'], 'lab': ['5', '6', '7', '8'], 'office': ['4', '5', '6']}

            for place in test_label_idx:
                for i, idx in enumerate(test_label_idx[place]):
                    Rframes = glob.glob(os.path.join(data_dir, "frames/test/%s/%s/Rhand/*.png"%(place, i+1)))
                    Lframes = glob.glob(os.path.join(data_dir, "frames/test/%s/%s/Lhand/*.png"%(place, i+1)))
                    Rframes.sort(key=lambda x:int(x.split('Image')[1].split('.')[0]))
                    Lframes.sort(key=lambda x:int(x.split('Image')[1].split('.')[0]))
                    if len(Rframes) != len(Lframes):
                        pdb.set_trace()
                    Rlabels = np.load(os.path.join(data_dir,'labels/%s/obj_right%s.npy'%(place, idx)))
                    Llabels = np.load(os.path.join(data_dir,'labels/%s/obj_left%s.npy'%(place, idx)))
                    for j in range(len(Rframes)):
                        self.data.append([Rframes[j], Rlabels[j]])
                        self.data.append([Lframes[j], Llabels[j]])
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if split == 'train':
            self.transform = transforms.Compose([transforms.Scale(256),
                                                 transforms.RandomSizedCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:
            self.transform = transforms.Compose([transforms.Scale(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 normalize])
        

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        image_path = os.path.join(self.data_dir, self.data[index][0])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.LongTensor([int(self.data[index][1])])
        return image, label

    def __len__(self):
        return len(self.data)

