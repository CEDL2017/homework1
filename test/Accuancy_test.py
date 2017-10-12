import _init_paths
from data_helper import readScoreNetData,readTestData
from scorenet import ScoreNet
from config import*

model = ScoreNet(save_path=model_path)
model.compile()
# model.train()
(_,accuracy) = model.evaluate()
print('test accuracy: ', accuracy)
