import subprocess, sys
'''
model_name = 'resnet18'
cmd = 'mkdir checkpoint/'+ model_name

out = subprocess.call(cmd, shell=True)
cmd = 'CUDA_VISIBLE_DEVICES=6 python main.py --batch-size 128 --workers 4 --pretrained  --model_name \''+model_name+'\'   2>&1 | tee checkpoint/'+model_name+'/log'
out = subprocess.call(cmd, shell=True)
'''

model_name = 'resnet50'
cmd = 'mkdir checkpoint/'+ model_name

out = subprocess.call(cmd, shell=True)
cmd = 'CUDA_VISIBLE_DEVICES=1 python main.py  --batch-size 128  --arch resnet50  --workers 4 --pretrained  --model_name \''+model_name+'\'   2>&1 | tee checkpoint/'+model_name+'/log'
out = subprocess.call(cmd, shell=True)

