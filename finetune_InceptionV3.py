# Ref: https://keras-cn.readthedocs.io/en/latest/other/application/
# Edit: Huiting Hong

from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
from datagenerator_non2string import ImageDataGenerator_custom
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.model_selection import train_test_split

import cv2
import numpy as np


datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=13.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False)

def image_augmentation(X_train, Y_train, batch_size, datagen=datagen):

    datagen.fit(X_train)

    # fits the model on batches with real-time data augmentation:
    return datagen.flow(X_train, Y_train, batch_size=batch_size).next()

def read_as_img(paths,batch_size):
    scale_size = (277,277)
    mean = np.array([104., 117., 124.])
    images = np.ndarray([batch_size, scale_size[0], scale_size[1], 3])
    for i in range(len(paths)):
        img = cv2.imread(paths[i])

        #rescale image
        img = cv2.resize(img, (scale_size[0], scale_size[1]))
        img = img.astype(np.float32)
        
        #subtract mean
        img -= mean
                                                             
        images[i] = img
    return images

def onehot(labels, n_classes, batch_size):
    # n_classes = 24
    one_hot_labels = np.zeros((batch_size, n_classes))
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1

    return one_hot_labels

def DataGenerator(X, y, bt_size, n_classes, data_aug=True):
    while 1:
        p = np.random.permutation(len(X))
        X, y = X[p],y[p]
        bt_index = len(X)//bt_size
        for i in range(bt_index):
            X_batch = X[i*bt_size:(i+1)*bt_size]
            y_batch = y[i*bt_size:(i+1)*bt_size]
            X_batch = read_as_img(X_batch,bt_size)
            y_batch = onehot(y_batch,n_classes,bt_size)

            if data_aug:
                X_batch, y_batch = image_augmentation(X_batch, y_batch, bt_size)
            else:
                X_batch = preprocess_input(X_batch)
            yield (X_batch, y_batch)



# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))

# Specify data directory
train_val_file = './hand_all_train.txt'
test_file = './hand_all_test.txt'

# Specify class#, epoch#, bt-size
n_classes = 24
epochs_toplayer = 5
epochs_InceptionAndToplayer = 10
batch_size = 16

# Create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(n_classes, activation='softmax')(x)

# The model we are going train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all convolutional InceptionV3 layers and train only top layers which we randomly initialize
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])#'categorical_crossentropy')


# Initalize the data generator seperately for the training and validation set
train_val_generator = ImageDataGenerator_custom(train_val_file)
test_generator = ImageDataGenerator_custom(test_file, shuffle = False)

train_val_img = np.asarray(train_val_generator.images)
train_val_label = np.asarray(train_val_generator.labels)

X_train, X_val, y_train, y_val = train_test_split(train_val_img, train_val_label, test_size=0.33)

# print('train_x length = ',len(X_train))
# print('val_x length = ',len(X_val))

# Train the model on the new data for a few epochs
checkpointer = ModelCheckpoint(filepath='./best_fstlayer_model.hdf5', verbose=1, save_best_only=True)
model.fit_generator (DataGenerator(X_train,y_train,batch_size,n_classes),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs_toplayer,
        validation_data=DataGenerator(X_val,y_val,batch_size,n_classes,False),
        nb_val_samples=len(X_val) // batch_size,    
        callbacks=[checkpointer])

print('finish finetune on the top layer!')


# load model trained on top-layers
# model = load_model('best_model_epo30.hdf5')


# Start fine-tuning convolutional layers from inception V3. 
# freeze the bottom 249 layers and train the remaining top layers.
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# recompile model
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])#'categorical_crossentropy')


checkpointer = ModelCheckpoint(filepath='./best_model_epo40.hdf5', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
              patience=3, min_lr=0.00001)
history = model.fit_generator(DataGenerator(X_train,y_train,batch_size,n_classes),
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs_InceptionAndToplayer,
        validation_data=DataGenerator(X_val,y_val,batch_size,n_classes),
        validation_steps=len(X_val) // batch_size,
        callbacks=[checkpointer,reduce_lr])#LossHistory()])

model = load_model('best_model_epo40.hdf5')
print('load best model for testing')

x_test_len = len(test_generator.images)
x_test_lst = np.asarray(test_generator.images)
x_test = read_as_img(x_test_lst,x_test_len)
y_test = onehot(np.asarray(test_generator.labels),n_classes,x_test_len)

batch_size = 50

y_pre = model.predict(x_test,batch_size=batch_size,verbose=1)
# y_pre = np.argmax(y_pre[0])


# y_test_tmp = np.argmax(y_test,axis=1)
# y_pre_tmp = np.argmax(y_pre,axis=1)
# acc_n = 0
# for i in range(len(y_pre_tmp)):
#     if y_pre_tmp[i] == y_test_tmp[i]:
#         acc_n += 1
# print('acc = ',acc_n/len(y_pre_tmp))

## Draw Precision Recall Curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import colors as mcolors

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_pre[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_pre[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_pre.ravel())
average_precision["micro"] = average_precision_score(y_test, y_pre,
                                                     average="micro")


# Plot Precision-Recall curve for each class and iso-f1 curves
# setup plot details
colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
color_lst = []
for key in colors_dict:
    color_lst.append(key)

# colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal','navy', 'turquoise', 'darkorange', 'cornflowerblue' ])
colors = cycle(color_lst[10:34])

plt.figure(figsize=(10, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i, color in zip(range(n_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(i, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0.5, -.38), prop=dict(size=14))

plt.show()

# loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
# print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

## list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


