import numpy
from numpy import array as ar
from scipy.ndimage import imread
from scipy.misc import imresize
import os

def frame_converter(size, fold):
    c = 0
    matrices = []
    for d in ['house', 'lab', 'office']:
        for i in range(3):
            for h in ['Lhand', 'Rhand']:
                images = os.listdir('E:/CEDL/' + fold + '/' + d + '/' + str(i + 1) + '/' + h + '/')
                for j in range(len(images)):
                    matrix = imread('E:/CEDL/' + fold + '/' + d + '/' + str(i + 1) + '/' + h + '/Image' + str(j + 1) + '.png', 'L')
                    matrix = imresize(matrix, size, 'nearest')
                    matrix = matrix / 255
                    matrices += [ar([matrix], dtype = 'float32')]
                    c += 1
                    if c % 100 == 0:
                        print(c)
    matrices = ar(matrices,  dtype = 'float32')
    numpy.save('E:/matrices_' + fold + '.npy', matrices)

def state_converter(kind, lenf, fold):
    c = 0
    arrays = []
    for d in ['house', 'lab', 'office']:
        if fold == 'train':
            rang = (0, 3)
        if fold == 'valid':
            rang = (3, 6)
        for i in range(rang[0], rang[1]):
            for h in ['left', 'right']:
                indices = numpy.load('E:/CEDL/labels/' + d + '/' + kind + '_' + h + str(i + 1) + '.npy')
                for index in indices:
                    array = numpy.zeros(lenf, dtype = 'int8')
                    array[int(index)] = 1
                    arrays += [array]
                    c += 1
                    if c % 100 == 0:
                        print(c)
    arrays = ar(arrays, dtype = 'int8')
    numpy.save('E:/arrays_' + fold + '_' + kind + '.npy', arrays)

frame_converter(size = (128, 228), fold = 'train')
frame_converter(size = (128, 228), fold = 'valid')
state_converter(fold = 'train', kind = 'ges', lenf = 13)
state_converter(fold = 'valid', kind = 'ges', lenf = 13)
state_converter(fold = 'train', kind = 'obj', lenf = 24)
state_converter(fold = 'valid', kind = 'obj', lenf = 24)
