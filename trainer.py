import numpy
from numpy import array as ar
from numpy.random import choice
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

def builder(input, drate, ouput):
    model = Sequential()
    model.add(Dropout(drate, input_shape = input))
    model.add(Conv2D( 16, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Dropout(drate))
    model.add(Conv2D( 32, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Dropout(drate))
    model.add(Conv2D( 64, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Dropout(drate))
    model.add(Conv2D(128, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Dropout(drate))
    model.add(Conv2D(256, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Dropout(drate))
    model.add(Conv2D(512, 3, padding = 'same', activation = 'elu', data_format = 'channels_first'))
    model.add(MaxPooling2D(2, 2, 'valid', data_format = 'channels_first'))
    model.add(Flatten())
    model.add(Dropout(drate))
    model.add(Dense(512, activation = 'elu'))
    model.add(Dropout(drate))
    model.add(Dense(512, activation = 'elu'))
    model.add(Dropout(drate))
    model.add(Dense(ouput, activation = 'softmax'))
    return model

def splitter(x, y, v):
    valid_choice = set(choice(len(x), size=int(len(x) * v), replace=False))
    train_choice = set(i for i in range(len(x))) - valid_choice
    x_train = []
    y_train = []
    for i in train_choice:
        x_train += [x[i]]
        y_train += [y[i]]
    x_valid = []
    y_valid = []
    for i in valid_choice:
        x_valid += [x[i]]
        y_valid += [y[i]]
    return ar(x_train, 'float32'), ar(y_train, 'int8'), ar(x_valid, 'float32'), ar(y_valid, 'int8')

# x, y, x_valid, y_valid = splitter(x, y, 0.25)

x_train = numpy.load('matrices_train.npy')
x_valid = numpy.load('matrices_valid.npy')
y_train = numpy.load('arrays_train_ges.npy')
y_valid = numpy.load('arrays_valid_ges.npy')

model = builder(x_train[0].shape, 0.1, y_train[0].shape[0])
optimizer = SGD(lr = 0.01, momentum = 0.9, decay = 0, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
history = model.fit(batch_size = 100, epochs = 10, verbose = 2, x = x_train, y = y_train, validation_data = (x_valid, y_valid), shuffle = True)
thist, vhist, accu = history.history['loss'], history.history['val_loss'], history.history['val_acc']
with open('saved_history', 'w') as saved_history:
    json.dunp([thist, vhist, accu], saved_history)
