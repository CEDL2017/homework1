import numpy as np 
import scipy as sp 
from scipy import misc
import tensorflow as tf 
import os
import re
from PIL import Image


def read_data():
    labels = np.array([])
    #first loading the train files, then the test files
    for i in range(2):
        for location in ['house', 'lab', 'office']:
            for hand in ['left', 'right']:            
                for j in range(3):
                    file_name = 'FA_' + hand + str(i * 3 + j + 1) + '.npy'
                    print(file_name)
                    try:
                        label_path = os.path.join('data/labels', location, file_name)
                        label_file = np.load(label_path)
                        labels = np.append(labels, label_file)
                    except:
                        print('error when loading: ', label_path)
    print(labels.shape) #25828
    
def load_data():
    frames = None
    #first loading the train files, then the test files
    for i in ['train', 'test']:
        for location in ['house', 'lab', 'office']:
            for hand in ['Lhand', 'Rhand']:            
                for j in range(3):                    
                    target_folder_path = os.path.join('data/frames', i, location, str(j + 1), hand)                    
                    print(target_folder_path)
                    file_names = os.listdir(target_folder_path)
                    file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
                    for name in file_names:                                                
                        try:                            
                            image = misc.imread(os.path.join(target_folder_path, name)).astype(int)
                            print(os.path.join(target_folder_path, name))
                            if not (frames is None):
                                frames = np.append(frames, image)                                
                            else:
                                frames = image                                                                                                                          
                        except:
                            print('error when loading: ', target_folder_path, name)
    print(len(frames)) #25828


def process_data():
    cnt = 0
    for i in ['train', 'test']:
        for location in ['house', 'lab', 'office']:
            for hand in ['Lhand', 'Rhand']:            
                for j in range(3):                    
                    target_folder_path = os.path.join('data/frames', i, location, str(j + 1), hand)                    
                    print(target_folder_path)
                    file_names = os.listdir(target_folder_path)
                    file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
                    for name in file_names:                                                
                        try:                            
                            image = Image.open(os.path.join(target_folder_path, name))
                            out = image.resize((227,227))
                            out.save(os.path.join('processed_data', i, 'Image' + str(cnt) + '.png'), 'PNG')
                            aug_out = out.transpose(Image.FLIP_LEFT_RIGHT)
                            aug_out.save(os.path.join('processed_data', i, 'aug_Image' + str(cnt) + '.png'), 'PNG')
                            cnt += 1
                        except:
                            print('error when loading: ', target_folder_path, name)

process_data()

# im = Image.open('data/frames/train/house/1/Lhand/Image100.png')

# out = im.resize((128,128))
# out2 = out.transpose(Image.FLIP_LEFT_RIGHT)
# #out.save('precessed_data/out.jpg')
# out.save('out.png', 'PNG')
# out2.save('out2.jpg', 'JPEG')
# out2.save('precessed_data/out2.jpg')

# load_data()  
# read_data()
# os.path.join()

# label_path = 'data/labels/house/FA_left1.npy'
# y = np.load(label_path)
# print(y)



# target_folder_path = 'data/frames/train/house/1/head'
# file_names = os.listdir(target_folder_path)
# file_names = sorted(file_names, key=lambda x: int(re.sub('\D', '', x)))
# print(file_names)



# train_x, train_y, test_x, text_y = load_data()

# logits = model(X)
# with tf.name_scope("loss"):
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="loss")    

# with tf.name_scope("train"):
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# with tf.name_scope("accuracy"):
#     correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     # tf.summary.scalar("accuracy", accuracy)


# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     sess.run(tf.global_variables_initializer())
    
#     for i in range(2001):
#         batch = mnist.train.next_batch(100)
#         sess.run(train_step, feed_dict={X: , y: })

#         if i % 5 == 0:
#             [loss, train_accuracy] = sess.run([loss, accuracy], feed_dict={x: batch[0], y: })
#             print('training loss:', loss)
#             print('training accuracy:', train_accuracy)
        