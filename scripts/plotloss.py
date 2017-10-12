# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 00:41:10 2017

@author: HGY
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
os.chdir('D:\Lab\CEDL\homework1\scripts')


#%% Paths
LOG_PATH = './resnet_twostring_16_noweighted_1conv.log'
SAVE_PATH = '../results/'+os.path.basename(LOG_PATH).split('.')[0]+'.png'


#%% Plot result
df = pd.read_csv(LOG_PATH, index_col=0)
trainAcc = df['acc'].tolist()
trainLoss = df['loss'].tolist()
testAcc = df['val_acc'].tolist()
testLoss = df['val_loss'].tolist()

fig = plt.figure()
pltloss = fig.add_subplot(121)
x_axis = np.arange(0,len(testAcc))                                                                       
pltloss.plot(x_axis,np.asarray(trainLoss), label='LossTrain')
pltloss.plot(x_axis, np.asarray(testLoss), label='LossTest')

plt.title('Cross Entropy Loss')
plt.xlabel("epoch-time") 
plt.ylabel("cross-entropy")
plt.grid(True)
plt.legend()

pltaccu = fig.add_subplot(122)
pltaccu.plot(x_axis, np.asarray(trainAcc), label='AccTrain')
pltaccu.plot(x_axis, np.asarray(testAcc), label='AccTest')

plt.xlabel("epoch") 
plt.ylabel("accuracy")
plt.title("Accuracy")
plt.grid(True)   
plt.legend()
plt.savefig(SAVE_PATH,dpi=300,format="png") 
