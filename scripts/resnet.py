# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:58:47 2017

@author: HGY
"""
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Input, Dense, concatenate, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.fixes import bincount
import numpy as np
import pandas as pd
import cv2
import imgaug
import os
os.chdir('D:\Lab\CEDL\hw1\scripts')


#%% Parameters
IMGW = 224
IMGH = 224
IMGC = 3
TRAINDATA = './hand_train.txt'
TESTDATA = './hand_test.txt'


#%% Build Fine-Tune Model
imgInput = Input((IMGW, IMGH, IMGC, ), name='image_input')
ResNet = ResNet50(weights='imagenet')
#ResFeaExtract = Model(inputs=ResNet.input, outputs=ResNet.get_layer('flatten_1').output)
#fineTuneList = ['res5c_branch2c', 'bn5c_branch2c', 'add_16', 'activation_49', 'avg_pool', 'flatten_1', 'fc1000']
fineTuneList = ['fc1000']
for layer in ResNet.layers:
    if layer.name in fineTuneList:
        continue
    layer.trainable = False
df = pd.DataFrame(([layer.name, layer.trainable] for layer in ResNet.layers), columns=['layer', 'trainable'])

flatten = ResNet(imgInput)
BatchNorm1 = BatchNormalization()(flatten)
dense1 = Dense(100, activation='relu', name='fine_tune_dense1')(BatchNorm1)
BatchNorm2 = BatchNormalization()(dense1)
output = Dense(24, activation='linear', name='model_output')(BatchNorm2)
classifier = Model(inputs=imgInput, outputs=output)       
classifier.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])


#%% Data Generator
def compute_class_weight(labelPath):
    with open(labelPath, 'r') as text_file:
        content = text_file.readlines()
    content = np.asarray(content)
    y = np.asarray([int(sample.split(' ')[1].strip('\n')) for sample in content] )
    
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
    
def myGenerator(dataPath, batchSize=100, shuffle=True, randseed=0, n_classes=24):
    with open(dataPath, 'r') as text_file:
        content = text_file.readlines()
    content = np.asarray(content)
        
    np.random.seed(randseed)
    dataSize = len(content)
    iterations = int(np.floor(dataSize / batchSize))
    
    while 1:
        if shuffle==True:
            dataIdx = np.random.permutation(dataSize)
        else:
            dataIdx = np.asarray(range(dataSize))
            
        for ii in range(iterations-1):
            idx = dataIdx[ii*batchSize:(ii+1)*batchSize]
            batchData = content[idx]
            imgArray = []
            labArray = []
            for sample in list(batchData):
                imgPath = sample.split(' ')[0]
                label = sample.split(' ')[1].strip('\n')
                imgArray.append(imgPreProcess(cv2.imread(imgPath)))
                labArray.append(label)
            imgArray = np.asarray(imgArray)
            labArray = np.asanyarray(labArray).astype('int')
            
            one_hot_labels = np.zeros((batchSize, n_classes))
            for i in range(len(labArray)):
                one_hot_labels[i][labArray[i]] = 1
            one_hot_labels = one_hot_labels.astype('float32')

            yield imgArray, one_hot_labels



#%% Train Model
TRAINSIZE = 14992
TESTSIZE = 15628
BATCHSIZE = 16
VERBOSE = 1
N_CLASS = 24
EPOCHS = 50
#classWeight = compute_class_weight(TRAINDATA)
trainGenerator = myGenerator(TRAINDATA, batchSize=BATCHSIZE, n_classes=N_CLASS)
testGenerator = myGenerator(TESTDATA, shuffle=False, batchSize=BATCHSIZE, n_classes=N_CLASS)

#tensorBoard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BATCHSIZE, write_graph=True, 
#                          write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, 
#                          embeddings_metadata=None)

filepath='../model/weigh-'+str(BATCHSIZE)+'-resnet-noweighted-fc-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('resnet_'+str(BATCHSIZE)+'_noweighted_fc.log') 

classifier.fit_generator(trainGenerator, 
                         samples_per_epoch=int(np.floor(TRAINSIZE/BATCHSIZE)), 
                         epochs = EPOCHS, 
                         verbose=VERBOSE,
                         class_weight = None,
                         validation_data = testGenerator,
                         validation_steps = int(np.floor(TESTSIZE/BATCHSIZE)),
                         workers = 1,
                         callbacks = [csv_logger, checkpoint])



