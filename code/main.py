from __future__ import print_function
from __future__ import division

import os
import pdb
import csv
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch import nn
from torch.autograd import Variable
from torch.optim import Adam

from resnet import resnet50 # homemade resnet pre-trained models :)
from resnet import resnet34 # homemade resnet pre-trained models :)

USE_GPU = True

TRAIN_HEAD_DIRS = ['../data/frames/train/house/1/head/', '../data/frames/train/house/1/head/',
                   '../data/frames/train/house/2/head/', '../data/frames/train/house/2/head/',
                   '../data/frames/train/house/3/head/', '../data/frames/train/house/3/head/',
                   '../data/frames/train/lab/1/head/', '../data/frames/train/lab/1/head/',
                   '../data/frames/train/lab/2/head/', '../data/frames/train/lab/2/head/',
                   '../data/frames/train/lab/3/head/', '../data/frames/train/lab/3/head/',
                   '../data/frames/train/lab/4/head/', '../data/frames/train/lab/4/head/',
                   '../data/frames/train/office/1/head/', '../data/frames/train/office/1/head/',
                   '../data/frames/train/office/2/head/', '../data/frames/train/office/2/head/',
                   '../data/frames/train/office/3/head/', '../data/frames/train/office/3/head/',]

TRAIN_HAND_DIRS = ['../data/frames/train/house/1/Lhand/', '../data/frames/train/house/1/Rhand/',
                   '../data/frames/train/house/2/Lhand/', '../data/frames/train/house/2/Rhand/',
                   '../data/frames/train/house/3/Lhand/', '../data/frames/train/house/3/Rhand/',
                   '../data/frames/train/lab/1/Lhand/', '../data/frames/train/lab/1/Rhand/',
                   '../data/frames/train/lab/2/Lhand/', '../data/frames/train/lab/2/Rhand/',
                   '../data/frames/train/lab/3/Lhand/', '../data/frames/train/lab/3/Rhand/',
                   '../data/frames/train/lab/4/Lhand/', '../data/frames/train/lab/4/Rhand/',
                   '../data/frames/train/office/1/Lhand/', '../data/frames/train/office/1/Rhand/',
                   '../data/frames/train/office/2/Lhand/', '../data/frames/train/office/2/Rhand/',
                   '../data/frames/train/office/3/Lhand/', '../data/frames/train/office/3/Rhand/',]

TEST_HEAD_DIRS = ['../data/frames/test/house/1/head/', '../data/frames/test/house/1/head/',
                  '../data/frames/test/house/2/head/', '../data/frames/test/house/2/head/',
                  '../data/frames/test/house/3/head/', '../data/frames/test/house/3/head/',
                  '../data/frames/test/lab/1/head/', '../data/frames/test/lab/1/head/',
                  '../data/frames/test/lab/2/head/', '../data/frames/test/lab/2/head/',
                  '../data/frames/test/lab/3/head/', '../data/frames/test/lab/3/head/',
                  '../data/frames/test/lab/4/head/', '../data/frames/test/lab/4/head/',
                  '../data/frames/test/office/1/head/', '../data/frames/test/office/1/head/',
                  '../data/frames/test/office/2/head/', '../data/frames/test/office/2/head/',
                  '../data/frames/test/office/3/head/', '../data/frames/test/office/3/head/',]

TEST_HAND_DIRS = ['../data/frames/test/house/1/Lhand/', '../data/frames/test/house/1/Rhand/',
                  '../data/frames/test/house/2/Lhand/', '../data/frames/test/house/2/Rhand/',
                  '../data/frames/test/house/3/Lhand/', '../data/frames/test/house/3/Rhand/',
                  '../data/frames/test/lab/1/Lhand/', '../data/frames/test/lab/1/Rhand/',
                  '../data/frames/test/lab/2/Lhand/', '../data/frames/test/lab/2/Rhand/',
                  '../data/frames/test/lab/3/Lhand/', '../data/frames/test/lab/3/Rhand/',
                  '../data/frames/test/lab/4/Lhand/', '../data/frames/test/lab/4/Rhand/',
                  '../data/frames/test/office/1/Lhand/', '../data/frames/test/office/1/Rhand/',
                  '../data/frames/test/office/2/Lhand/', '../data/frames/test/office/2/Rhand/',
                  '../data/frames/test/office/3/Lhand/', '../data/frames/test/office/3/Rhand/',]

TRAIN_FA_LABELS = ['../data/labels/house/FA_left1.npy', '../data/labels/house/FA_right1.npy',
                   '../data/labels/house/FA_left2.npy', '../data/labels/house/FA_right2.npy',
                   '../data/labels/house/FA_left3.npy', '../data/labels/house/FA_right3.npy',
                   '../data/labels/lab/FA_left1.npy', '../data/labels/lab/FA_right1.npy',
                   '../data/labels/lab/FA_left2.npy', '../data/labels/lab/FA_right2.npy',
                   '../data/labels/lab/FA_left3.npy', '../data/labels/lab/FA_right3.npy',
                   '../data/labels/lab/FA_left4.npy', '../data/labels/lab/FA_right4.npy',
                   '../data/labels/office/FA_left1.npy', '../data/labels/office/FA_right1.npy',
                   '../data/labels/office/FA_left2.npy', '../data/labels/office/FA_right2.npy',
                   '../data/labels/office/FA_left3.npy', '../data/labels/office/FA_right3.npy',]

TRAIN_OBJ_LABELS = ['../data/labels/house/obj_left1.npy', '../data/labels/house/obj_right1.npy',
                     '../data/labels/house/obj_left2.npy', '../data/labels/house/obj_right2.npy',
                     '../data/labels/house/obj_left3.npy', '../data/labels/house/obj_right3.npy',
                     '../data/labels/lab/obj_left1.npy', '../data/labels/lab/obj_right1.npy',
                     '../data/labels/lab/obj_left2.npy', '../data/labels/lab/obj_right2.npy',
                     '../data/labels/lab/obj_left3.npy', '../data/labels/lab/obj_right3.npy',
                     '../data/labels/lab/obj_left4.npy', '../data/labels/lab/obj_right4.npy',
                     '../data/labels/office/obj_left1.npy', '../data/labels/office/obj_right1.npy',
                     '../data/labels/office/obj_left2.npy', '../data/labels/office/obj_right2.npy',
                     '../data/labels/office/obj_left3.npy', '../data/labels/office/obj_right3.npy',]

TEST_FA_LABELS = ['../data/labels/house/FA_left4.npy', '../data/labels/house/FA_right4.npy',
                  '../data/labels/house/FA_left5.npy', '../data/labels/house/FA_right5.npy',
                  '../data/labels/house/FA_left6.npy', '../data/labels/house/FA_right6.npy',
                  '../data/labels/lab/FA_left5.npy', '../data/labels/lab/FA_right5.npy',
                  '../data/labels/lab/FA_left6.npy', '../data/labels/lab/FA_right6.npy',
                  '../data/labels/lab/FA_left7.npy', '../data/labels/lab/FA_right7.npy',
                  '../data/labels/lab/FA_left8.npy', '../data/labels/lab/FA_right8.npy',
                  '../data/labels/office/FA_left4.npy', '../data/labels/office/FA_right4.npy',
                  '../data/labels/office/FA_left5.npy', '../data/labels/office/FA_right5.npy',
                  '../data/labels/office/FA_left6.npy', '../data/labels/office/FA_right6.npy',]

TEST_OBJ_LABELS = ['../data/labels/house/obj_left4.npy', '../data/labels/house/obj_right4.npy',
                    '../data/labels/house/obj_left5.npy', '../data/labels/house/obj_right5.npy',
                    '../data/labels/house/obj_left6.npy', '../data/labels/house/obj_right6.npy',
                    '../data/labels/lab/obj_left5.npy', '../data/labels/lab/obj_right5.npy',
                    '../data/labels/lab/obj_left6.npy', '../data/labels/lab/obj_right6.npy',
                    '../data/labels/lab/obj_left7.npy', '../data/labels/lab/obj_right7.npy',
                    '../data/labels/lab/obj_left8.npy', '../data/labels/lab/obj_right8.npy',
                    '../data/labels/office/obj_left4.npy', '../data/labels/office/obj_right4.npy',
                    '../data/labels/office/obj_left5.npy', '../data/labels/office/obj_right5.npy',
                    '../data/labels/office/obj_left6.npy', '../data/labels/office/obj_right6.npy',]

SCENES = ['house', 'lab', 'office']

EPOCH = 300
BATCH_SIZE = 64

MSG_DISPLAY_FREQ = 20

LOSS = np.array([])

class HandcamDataset:

    def __init__(self, head_dirs, hand_dirs, fa_nps, obj_nps, transform=None):
        # check the input params
        assert len(head_dirs) == len(hand_dirs) == len(fa_nps) == len(obj_nps)
        # retrieve all the filenames
        self.data = []
        for (head_dir, hand_dir, fa_np, obj_np) in zip(head_dirs, hand_dirs, fa_nps, obj_nps):
            fa_labels = np.load(fa_np)
            obj_labels = np.load(obj_np)
            head_filenames = os.listdir(head_dir)
            head_filenames = sorted(head_filenames,
                                    key=lambda pid: int(pid.split('Image')[1].split('.')[0]))
            hand_filenames = os.listdir(hand_dir)
            hand_filenames = sorted(hand_filenames,
                                    key=lambda pid: int(pid.split('Image')[1].split('.')[0]))
            for (head_filename, hand_filename, fa_label, obj_label) in zip(head_filenames, hand_filenames, fa_labels, obj_labels):
                self.data.append({'head_filename':head_dir+head_filename,
                                  'hand_filename':hand_dir+hand_filename,
                                  'fa_label':fa_label,
                                  'obj_label':obj_label},)
        # store transformation settings
        self.transform = transform

    def __getitem__(self, index):
        scene = SCENES.index(self.data[index]['head_filename'].split('/')[4])
        head_img = Image.open(self.data[index]['head_filename'])
        head_img = head_img.convert('RGB')
        head_img = head_img.resize((224, 224), resample=Image.LANCZOS)
        hand_img = Image.open(self.data[index]['hand_filename'])
        hand_img = hand_img.convert('RGB')
        hand_img = hand_img.resize((224, 224), resample=Image.LANCZOS)
        if self.transform is not None:
            head_img = self.transform(head_img)
            hand_img = self.transform(hand_img)
        assert isinstance(head_img, torch.FloatTensor)   # img must be torch.FloatTensor
        assert isinstance(hand_img, torch.FloatTensor)   # img must be torch.FloatTensor
        fa_label = torch.LongTensor([long(self.data[index]['fa_label'])])  # label must be torch.LongTensor
        obj_label = torch.LongTensor([long(self.data[index]['obj_label'])])  # label must be torch.LongTensor
        return scene, head_img, hand_img, fa_label, obj_label

    def __len__(self):
        return len(self.data)


class HanNet(nn.Module):

    def __init__(self, pretrained=False, num_classes=1000):
        super(HanNet, self).__init__()
        self.headstream = resnet34(pretrained=pretrained)
        self.handstream = resnet34(pretrained=pretrained)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x = torch.cat((self.headstream(x[0]), self.handstream(x[1])), dim=1)
        x = self.fc(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = 0.0

    # switch to train mode
    model.train()

    end = time.time()

    running_loss = 0.0
    global LOSS

    for i, (_, head_inputs, hand_inputs, fa_labels, obj_labels) in enumerate(train_loader):

        fa_labels = torch.squeeze(fa_labels, 1)
        obj_labels = torch.squeeze(obj_labels, 1)

        if USE_GPU:
            head_inputs = Variable(head_inputs).cuda(async=True)
            hand_inputs = Variable(hand_inputs).cuda(async=True)
            fa_labels = Variable(fa_labels).cuda(async=True)
            obj_labels = Variable(obj_labels).cuda(async=True)
        else:
            head_inputs = Variable(head_inputs)
            hand_inputs = Variable(hand_inputs)
            fa_labels = Variable(fa_labels)
            obj_labels = Variable(obj_labels)
        
        outputs = model([head_inputs, hand_inputs])

        loss = criterion(outputs[:, 0:2], fa_labels) + criterion(outputs[:, 2:26], obj_labels) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        batch_time += time.time() - end
        end = time.time()

        if i%MSG_DISPLAY_FREQ == (MSG_DISPLAY_FREQ-1):
            print("Epoch: [{}][{}/{}]\t Loss: {:.8f}\t Time {:.3f}".format(epoch, i+1, len(train_loader), running_loss/MSG_DISPLAY_FREQ, batch_time/MSG_DISPLAY_FREQ))
            LOSS = np.append(LOSS, running_loss/MSG_DISPLAY_FREQ)
            running_loss = 0.0
            batch_time = 0.0

    np.save('loss_{:03}.npy'.format(epoch), LOSS)


def test(test_loader, model, epoch):

    # switch to evaluate mode
    model.eval()

    confusion_matrix = np.zeros((3, 24, 24))

    for i, (scenes, head_inputs, hand_inputs, _, obj_labels) in enumerate(test_loader):

        obj_labels = torch.squeeze(obj_labels, 1)

        if USE_GPU:
            head_inputs = Variable(head_inputs, volatile=True).cuda(async=True)
            hand_inputs = Variable(hand_inputs, volatile=True).cuda(async=True)
            obj_labels = Variable(obj_labels, volatile=True).cuda(async=True)
        else:
            head_inputs = Variable(head_inputs, volatile=True)
            hand_inputs = Variable(hand_inputs, volatile=True)
            obj_labels = Variable(obj_labels, volatile=True)
        
        outputs = model([head_inputs, hand_inputs])

        _, predictions = torch.max(outputs[:, 2:25], 1)

        for j in range(predictions.data.size(0)):

            scene = scenes[j]
            prediction = predictions.data[j]
            label = obj_labels.data[j]

            confusion_matrix[scene][prediction][label] += 1

    print("Acc: {:.3}".format(np.sum(np.trace(confusion_matrix, axis1=1, axis2=2))/np.sum(confusion_matrix)))
    np.save('cm_{:03}.npy'.format(epoch), confusion_matrix)

def main():

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HandcamDataset(TRAIN_HEAD_DIRS, TRAIN_HAND_DIRS, TRAIN_FA_LABELS, TRAIN_OBJ_LABELS, transformations)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = HandcamDataset(TEST_HEAD_DIRS, TEST_HAND_DIRS, TRAIN_FA_LABELS, TRAIN_OBJ_LABELS, transformations)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    print("=> using pre-trained model HanNet")
    model = HanNet(pretrained=True, num_classes=2+24)

    if USE_GPU:
        model = model.cuda()

    if USE_GPU:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCH):
        # train(train_loader, model, criterion, optimizer, epoch)
        test(test_loader, model, epoch)
        torch.save(model, 'model_{:03}.pth'.format(epoch))

if __name__ == '__main__':
    main()
