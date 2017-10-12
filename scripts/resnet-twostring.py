# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:23:04 2017

@author: HGY
"""

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Input, Dense, concatenate, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras.layers.merge import concatenate
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.fixes import bincount
import numpy as np
import pandas as pd
import cv2
import imgaug
import os
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


os.chdir('D:\Lab\CEDL\hw1\scripts')


#%% Parameters
IMGW = 224
IMGH = 224
IMGC = 3
TRAINDATA = './hand_head_train.txt'
TESTDATA = './hand_head_test.txt'


#%% Build Fine-Tune Model
inputHand = Input((IMGW, IMGH, IMGC, ), name='inputHand')
inputHead = Input((IMGW, IMGH, IMGC, ), name='inputHead')
ResNetHand = ResNet50(weights='imagenet')
ResNetHead = ResNet50(weights='imagenet')
#fineTuneList = ['res5c_branch2c', 'bn5c_branch2c', 'add_16', 'activation_49', 'avg_pool', 'flatten_1', 'fc1000']
fineTuneList = ['fc1000']
def alterTrainable(model, fineTuneList):
    for layer in model.layers:
        if layer.name in fineTuneList:
            continue
        layer.trainable = False
    return model    

ResNetHand = alterTrainable(ResNetHand, fineTuneList)
ResNetHead = alterTrainable(ResNetHead, fineTuneList)
df = pd.DataFrame(([layer.name, layer.trainable] for layer in ResNetHand.layers), columns=['layer', 'trainable'])

flatHand = ResNetHand(inputHand)
flatHead = ResNetHand(inputHead)

batchNormHand = BatchNormalization()(flatHand)
batchNormHead = BatchNormalization()(flatHead)
concat = concatenate([batchNormHand, batchNormHead])

dense1 = Dense(1000, activation='relu', name='fine_tune_dense1')(concat)
batchNorm1 = BatchNormalization()(dense1)
dense2 = Dense(100, activation='relu', name='fine_tune_dense2')(batchNorm1)
batchNorm2 = BatchNormalization()(dense2)
output = Dense(24, activation='linear', name='model_output')(batchNorm2)
classifier = Model(inputs=[inputHand, inputHead] , outputs=output)       
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


#%% Data Generator
def compute_class_weight(labelPath):
    with open(labelPath, 'r') as text_file:
        content = text_file.readlines()
    content = np.asarray(content)
    y = np.asarray([int(sample.split(' ')[2].strip('\n')) for sample in content] )
    
    classes = np.asarray(list(set(y)))
    le = LabelEncoder()
    y_ind = le.fit_transform(y)
    recip_freq = len(y) / (len(le.classes_) * bincount(y_ind).astype(np.float64))
    weight = recip_freq[le.transform(classes)]
    return weight

def imgPreProcess(img):
    img = img.astype('float32')
    img = cv2.resize(img, (224,224))
    img = (img-128)/128
    return img

def imgAugmentation(imgs):
    seq = iaa.Sequential([
            #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(1), # horizontally flip 50% of the images
            #iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
            ])
    return seq.augment_images(imgs)  
  
def myGenerator(dataPath, batchSize=32, mode='Train', shuffle=True, randseed=0, n_classes=24):
    with open(dataPath, 'r') as text_file:
        content = text_file.readlines()
    content = np.asarray(content)
        
    np.random.seed(randseed)
    dataSize = len(content)
    iterations = int(np.floor(dataSize / batchSize))
    sampleSize = batchSize
    
    if mode !='Train':
        shuffle = False
        
    while 1:
        if shuffle==True:
            dataIdx = np.random.permutation(dataSize)
        else:
            dataIdx = np.asarray(range(dataSize))
            
        for ii in range(iterations-1):
            idx = dataIdx[ii*batchSize:(ii+1)*batchSize]
            batchData = content[idx]
            imgHandArray = []
            imgHeadArray = []
            labArray = []
            for sample in list(batchData):
                imgHandPath = sample.split(' ')[0]
                imgHeadPath = sample.split(' ')[1]
                label = sample.split(' ')[2].strip('\n')
                imgHandArray.append(imgPreProcess(cv2.imread(imgHandPath)))
                imgHeadArray.append(imgPreProcess(cv2.imread(imgHeadPath)))
                labArray.append(label)
                
            # Data augmentation when training
            if mode =='Train':
                imgHandArrayFliplr = imgAugmentation(imgHandArray)
                imgHeadArrayFliplr = imgAugmentation(imgHeadArray)
                imgHandArray.extend(imgHandArrayFliplr)
                imgHeadArray.extend(imgHeadArrayFliplr)
                labArray.extend(labArray)
                sampleSize = batchSize*2
            imgHandArray = np.asarray(imgHandArray)
            imgHeadArray = np.asarray(imgHeadArray)
            labArray = np.asanyarray(labArray).astype('int')
            
            
            one_hot_labels = np.zeros((sampleSize, n_classes))
            for i in range(len(labArray)):
                one_hot_labels[i][labArray[i]] = 1
            one_hot_labels = one_hot_labels.astype('float32')

            yield {'inputHand':imgHandArray, 'inputHead':imgHeadArray}, one_hot_labels



#%% Train Model
TRAINSIZE = 14992
TESTSIZE = 15628
BATCHSIZE = 16
VERBOSE = 1
N_CLASS = 24
EPOCHS = 10
classWeight = compute_class_weight(TRAINDATA)
trainGenerator = myGenerator(TRAINDATA, mode='Train', batchSize=BATCHSIZE, n_classes=N_CLASS)
testGenerator = myGenerator(TESTDATA, mode='Test', shuffle=False, batchSize=BATCHSIZE, n_classes=N_CLASS)


#tensorBoard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCHSIZE, write_graph=True, 
#                          write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
#                          embeddings_metadata=None)

# checkpoint
filepath='../model/weigh-resnet-twostring-'+str(BATCHSIZE)+'-weighted-fc-fliplr-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('resnet_twostring_'+str(BATCHSIZE)+'_noweighted_fc_fliplr.log') 

classifier.fit_generator(trainGenerator, 
                         samples_per_epoch=int(np.floor(TRAINSIZE/BATCHSIZE)), 
                         epochs = EPOCHS, 
                         verbose=VERBOSE,
                         class_weight = classWeight,
                         validation_data = testGenerator,
                         validation_steps = int(np.floor(TESTSIZE/BATCHSIZE)),
                         workers = 1,
                         callbacks = [csv_logger, checkpoint])


#%% Load the best fc parameters then tune more layer
#classifier.load_weights('../model/weigh-resnet-twostring-32-weighted-fc-fliplr-27-0.51.hdf5')
#
#fineTuneList = ['res5c_branch2c', 'bn5c_branch2c', 'add_16', 'activation_49', 'avg_pool', 'flatten_1', 'fc1000']
#ResNetHand = alterTrainable(ResNetHand, fineTuneList)
#ResNetHead = alterTrainable(ResNetHead, fineTuneList)
#df = pd.DataFrame(([layer.name, layer.trainable] for layer in classifier.layers), columns=['layer', 'trainable'])
#
