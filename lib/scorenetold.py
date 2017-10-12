from keras.layers import Dense, Dropout, Activation, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import backend as K
from keras import metrics
from config import *
import os
from data_helper import readScoreNetData,readTestData
import numpy as np

K.set_image_dim_ordering('tf')

class ScoreNet(object):
    save_path = None
    def __init__(self, save_path='47.h5', use_VGG=False):
        self.model = Sequential()
        self.VGG = use_VGG
        if self.VGG:
            print("Using VGG16 net")
            print(scorenet_fc_num)
        else:
            print("Using VGG19 net")
            print(scorenet_fc_num)
            
        
        if not use_VGG:
            # 1st Conv
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_1", activation="relu",input_shape=(200, 200, 3)))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(16, (3, 3), padding='same',name="conv2_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(16, (3, 3), padding='same', name="conv2_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_3", activation="relu"))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_3", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_3", activation="relu"))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_4", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(Flatten(name="flatten"))
            self.model.add(Dense(512, activation='relu', name='dense_1'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu', name='dense_2'))
            self.model.add(Dropout(0.5))
            #self.model.add(Dense(scorenet_fc_num, name='dense_3'))
            #self.model.add(Activation('relu',name="final_layer"))
            self.model.add(Dense(scorenet_fc_num,activation='softmax',name='final_layer'))
        
        else:
            # self.model.add(ZeroPadding2D((1,1),input_shape=(270, 480, 3)))
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_1", activation="relu",input_shape=(270, 480, 3)))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(8, (3, 3), padding='same', name="conv1_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(16, (3, 3), padding='same',name="conv2_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(16, (3, 3), padding='same', name="conv2_2", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(32, (3, 3), padding='same', name="conv3_3", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv4_3", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_1", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_2", activation="relu"))
            # self.model.add(ZeroPadding2D((1,1)))
            self.model.add(Conv2D(64, (3, 3), padding='same', name="conv5_3", activation="relu"))
            self.model.add(MaxPooling2D((2,2), strides=(2,2)))

            self.model.add(Flatten(name="flatten"))
            self.model.add(Dense(512, activation='relu', name='dense_1'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(scorenet_fc_num, name='dense_3'))
            self.model.add(Activation("relu",name="final_layer"))

        # Load model if exist
        self.save_path = save_path
        if os.path.exists(save_path):
            print ("<< ScoreNet >> load score net pre-trained model...")
            self.model = load_model(save_path)
        print ("<< ScoreNet >> done initialize...")
        
    def compile(self):
        self.model.compile(
            
            loss='categorical_crossentropy',
            optimizer='adamax',
            metrics=[ metrics.categorical_accuracy]
        )

    def train(self):
        print(scorenet_fc_num)
        x_train = []
        y_train = []
        imgnum = 0
        for i in range(0,20):
                (x_batch, y_batch ,num) = readScoreNetData(i)
                imgnum +=num
                x_train = np.append(x_train,x_batch)
                y_train = np.append(y_train,y_batch)
                print ('FreeCount:',imgnum)
        x_train = x_train.reshape(-1,200,200,3)
        y_train = y_train.reshape(-1,24)
        x_train.astype('float32')
        y_train.astype('float32')
        print ('xtrain[i]:\n',x_train.shape)
        print ('xtrain[0]:\n',x_train[0])

        print ('ytrain[i]:\n',y_train.shape)
        # for e in range(50):
        #     for i in range(0,20):
        #         # (x_train, y_train) = readScoreNetData(i)
        #         # x_train.astype('float32')
        #         # y_train.astype('float32')
        #         print ('xtrain[i]:\n',x_train[i].shape)
        #         # print ('xtrain:\n',x_train[1])
        #         print ('ytrain[i]:\n',y_train[i].shape)
        self.model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1,validation_split=0.1)
        # self.model.fit(x, y, batch_size=, epochs=general_epoch, verbose=1)
        self.model.save(self.save_path)
        
        # print(self.model.evaluate(x_train,y_train))

    def test(self, x):
        return self.model.predict(x, batch_size=1, verbose=0)
    
    def evaluate(self):
        x_test = []
        y_test = []
        imgnum = 0
        for i in range(0,20):
                (x_batch, y_batch ,num) = readTestData(i)
                imgnum +=num
                x_test = np.append(x_test,x_batch)
                y_test = np.append(y_test,y_batch)
                print ('FreeCount:',imgnum)
        x_test = x_test.reshape(-1,200,200,3)
        y_test = y_test.reshape(-1,24)
        x_test.astype('float32')
        y_test.astype('float32')
        print ('xtrain[i]:\n',x_test.shape)
        print ('xtrain[0]:\n',x_test[0])
        return self.model.evaluate(x_test,y_test, verbose=1)

    
