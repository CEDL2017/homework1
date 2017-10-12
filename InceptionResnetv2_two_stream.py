
# coding: utf-8

# In[1]:


#parameter
fc_epochs = 10
epochs = 30
batch_size = 16
num_classes = 24
im_row = 200
im_col = 260
depth = 3
dropout_rate = 0.5


# In[ ]:


#model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import Model
from keras.layers.merge import Concatenate

hand_model = InceptionResNetV2(include_top=False,
                               input_shape=(im_row, im_col, depth),
                               name = 'hand')
hand_x = GlobalAveragePooling2D()(hand_model.output)
hand_x = Dropout(dropout_rate)(hand_x)

head_model = Xception(include_top=False,
                               input_shape=(im_row, im_col, depth))
head_x = GlobalAveragePooling2D()(head_model.output)
head_x = Dropout(dropout_rate)(head_x)

meta_input = Input(shape=(5,))
merge = Concatenate()([hand_x, head_x, meta_input])
x = Dense(1538, activation="relu")(merge)
x = Dropout(dropout_rate)(x)
out = Dense(num_classes, activation="softmax")(x)
model = Model([hand_model.input, head_model.input, meta_input], out)
model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])


# In[ ]:

#read data
from util import *

X_train, y_train, h_train, meta_train, X_val, y_val, h_val, meta_val = read_train_data_list(num_classes)
X_test, y_test, h_test, meta_test = read_test_data_list(num_classes)
print('Data complete!')

#data gen
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

def image_augmentation(X_train, h_train, y_train, batch_size, datagen=datagen):
    X_train, y_train = datagen.flow(X_train,
                                    y_train,
                                    batch_size=batch_size).next()
    h_train = datagen.flow(h_train,
                           batch_size=batch_size).next()

    return X_train, h_train, y_train

def data_generator(X, y, h, meta, batch_size, data_augmentation=True):
    while 1:
        p = np.random.permutation(len(X))
        X, y, h, meta = X[p], y[p], h[p], meta[p]
        batch_index = len(X)//batch_size
        for i in range(0, batch_index):
            X_batch =  X[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]
            h_batch = h[i*batch_size:(i+1)*batch_size]
            meta_batch = meta[i*batch_size:(i+1)*batch_size]
            X_batch = read_batch_data(X_batch)
            h_batch = read_batch_data(h_batch)
            
            if data_augmentation:
                X_batch, h_batch, y_batch = image_augmentation(X_batch, h_batch, y_batch, batch_size)
            else:
                X_batch = preprocess_input(X_batch)
                h_batch = preprocess_input(h_batch)
            yield ([X_batch, h_batch, meta_batch], y_batch)

# In[ ]:

from keras.callbacks import ModelCheckpoint, Callback

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.loss=[]
        self.val_loss=[]
    def on_epoch_end(self,epoch,logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
checkpoint = ModelCheckpoint('model', verbose=1, save_best_only=True)
loss_history = LossHistory()        
history = model.fit_generator(data_generator(X_train, y_train, h_train, meta_train, batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=data_generator(X_val, y_val, h_val, meta_val, batch_size, False),
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[loss_history, checkpoint])

# In[ ]:
#evaluate
"""
batch_size = 200
batch_index = len(X_test)//batch_size
acc = 0.0
for i in range(0, batch_index):
    X_batch =  X_test[i*batch_size:(i+1)*batch_size]
    y_batch = y_test[i*batch_size:(i+1)*batch_size]
    h_batch = h_test[i*batch_size:(i+1)*batch_size]
    meta_batch = meta_test[i*batch_size:(i+1)*batch_size]
    X_batch = read_batch_data(X_batch)
    X_batch = preprocess_input(X_batch)
    h_batch = read_batch_data(h_batch)
    h_batch = preprocess_input(h_batch)

    preds = model.predict([X_batch, h_batch, meta_batch])
    acc += np.sum(y_batch==np.argmax(preds,1))
    
X_batch =  X_test[i*batch_size:-1]
y_batch = y_test[i*batch_size:-1]
h_batch = h_test[i*batch_size:-1]
X_batch = read_batch_data(X_batch)
X_batch = preprocess_input(X_batch)
h_batch = read_batch_data(h_batch)
h_batch = preprocess_input(h_batch)
preds = model.predict([X_batch, h_batch, meta_batch])
acc += np.sum(y_batch==np.argmax(preds,1))
acc/=len(X_test)
print(acc)


from keras.models import load_model
model = load_model('model')

batch_size = 32
# In[ ]:



history = model.fit_generator(data_generator(X_train, y_train, batch_size),
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=data_generator(X_val, y_val, batch_size, False),
                    validation_steps=len(X_val) // batch_size,
                    callbacks=[loss_history, checkpoint])

batch_size = 500
batch_index = len(X_test)//batch_size
acc = 0.0
for i in range(0, batch_index):
    X_batch =  X_test[i*batch_size:(i+1)*batch_size]
    y_batch = y_test[i*batch_size:(i+1)*batch_size]
    X_batch = read_batch_data(X_batch)
    X_batch = preprocess_input(X_batch)
    preds = model.predict(X_batch)
    acc += np.sum(y_batch==np.argmax(preds,1))
X_batch =  X_test[i*batch_size:-1]
y_batch = y_test[i*batch_size:-1]
X_batch = read_batch_data(X_batch)
X_batch = preprocess_input(X_batch)
preds = model.predict(X_batch)
acc += np.sum(y_batch==np.argmax(preds,1))
acc/=len(X_test)
print(acc)
"""