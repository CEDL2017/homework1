import tensorflow as tf
import numpy as np
import os
import fnmatch

#%%

# you need to change this to your data directory
data_dir = '/home/viplab/Desktop/Syneh/HW01/data/'
os.chdir(data_dir)

def read_files(data_dir, is_train):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''

    image_list = []
    label_list = []
    with tf.name_scope('input'):
        if is_train:
            _dir = 'frames/train'
            num1 = '[1234]'
            num2 = '[123]'
        else:
            _dir = 'frames/test'
            num1 = '[5678]'
            num2 = '[456]'
        for root, dirs, files in os.walk(_dir):

            files.sort( key=lambda x:int(x[5:-4]))            
            for f in files:
                image_list.append(os.path.join(root,f))
                #print os.path.join(root,f)

        for root, dirs, files in os.walk('labels/'):
            files.sort()
            seq = ''
            if root == 'labels/lab' :
                seq = 'obj_*'+num1+'.npy'
            else:
                seq = 'obj_*'+num2+'.npy'
            for f in fnmatch.filter(files, seq):   
                hand = np.load(os.path.join(root,f))
                label_list += [int(i) for i in hand]
                #print os.path.join(root,f)
    print(image_list)
    print("Image_List Len::::",len(image_list),"Label Len::::::", len(label_list) )
    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_png(image_contents, channels=3)
    
    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
    # if you want to test the generated batches of images, you might want to comment the following line.

    image = tf.image.per_image_standardization(image)
    
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=batch_size,
#                                                      num_threads=64,
#                                                      capacity=capacity,
#                                                      min_after_dequeue=2000)
    print("Batches Done ?")
#    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    
    n_classes = 24
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])

    return image_batch, label_batch


