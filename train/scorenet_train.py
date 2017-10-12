import _init_paths
from data_helper import readScoreNetData,readTestData
from scorenet import ScoreNet
from config import*

# Read data
# while True:
# (x_train, y_train) = readScoreNetData() 
model = ScoreNet(save_path=model_path)
# x_train.astype('float32')
# y_train.astype('float32')
# print ('xtrain:\n',x_train.shape)
# print ('xtrain:\n',x_train[0])

# print ('ytrain:\n',y_train)
# break;
# Train
model.compile()
model.train()
# (_,accuracy) = model.evaluate()
# print('test accuracy: ', accuracy)
