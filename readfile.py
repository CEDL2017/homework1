import input_data
import tensorflow as tf
import numpy as np
import skimage.io as io
from PIL import Image
import os
#%%
def read_file():
    '''return image_list, , obj_label_list'''
    image_list = []
    obj_label_list = []
    place_name = { 0:"house", 1:"lab", 2:"office" }
    
    i = 0
    j = 1
    key = 0
    is_leftHand = True
    
    #house  
    while i <= 19:#house obj
    
        if is_leftHand : #左手
            
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_left'+str(j)+'.npy'
            train_path = '/home/viplab/Downloads/frames/train/'+place_name[key]+'/'+str(j)+'/Lhand/'
        else :  #右手    
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_right'+str(j)+'.npy'
            train_path = '/home/viplab/Downloads/frames/train/'+place_name[key]+'/'+str(j)+'/Rhand/'
                    
            
        #建立判斷obj的list
        imageNumber, temp_label_list = input_data.get_labe_Files(obj_label_path)
        obj_label_list+=temp_label_list
        #print(i,'obj: ',imageNumber)

        #建立frame的list
        temp_image_list = input_data.get_image_Files( train_path, imageNumber )
        image_list += temp_image_list
        #print(i,'image: ',len(temp_image_list))
        

        i = i+1
        j = j+1
        #場所替換
        if i == 6 or i ==14:
            key = key+1
  
        if place_name[key] == "lab":
           if j >4:
               j = 1
               if is_leftHand:
                   is_leftHand = False
               else :
                   is_leftHand = True 
        else:
            if j >3:
                j = 1
                if is_leftHand:
                    is_leftHand = False
                else :
                    is_leftHand = True
    
    temp = np.array([image_list, obj_label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    obj_label_list = list(temp[:,1])
    obj_label_list = [int(float(i)) for i in obj_label_list]
    return image_list, obj_label_list
#%%
def read_test():
    '''return image_list, obj_label_list '''
    image_list = []
    obj_label_list = []

    place_name = { 0:"house", 1:"lab", 2:"office" }
    index=[[3,4,5,6],[4,5,6,7,8],[3,4,5,6]]
    i = 0
    j = 1
    key = 0
    
    is_leftHand = True
    
  
    while i <= 19:#house obj
    
        if is_leftHand : #左手
            
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_left'+str(index[key][j])+'.npy'
            train_path = '/home/viplab/Downloads/frames/test/'+place_name[key]+'/'+str(j)+'/Lhand/'

        else :  #右手
            obj_label_path = '/home/viplab/Downloads/labels/'+place_name[key]+'/obj_right'+str(index[key][j])+'.npy'
            train_path = '/home/viplab/Downloads/frames/test/'+place_name[key]+'/'+str(j)+'/Rhand/'
            
                 
        #建立判斷obj的list
        imageNumber, temp_label_list = input_data.get_labe_Files(obj_label_path)
        obj_label_list+=temp_label_list
        

        #建立frame的list
        temp_image_list = input_data.get_image_Files( train_path, imageNumber )
        image_list += temp_image_list
        #print(i,'image: ',len(temp_image_list))
        

        i = i+1
        j = j+1
        #場所替換
        if i == 6 or i ==14:
            key = key+1
  
        if place_name[key] == "lab":
           if j >4:
               j = 1
               if is_leftHand:
                   is_leftHand = False
               else :
                   is_leftHand = True 
        else:
            if j >3:
                j = 1
                if is_leftHand:
                    is_leftHand = False
                else :
                    is_leftHand = True
    
    print(len(image_list)) 
    print(len(obj_label_list))

    temp = np.array([image_list, obj_label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    obj_label_list = list(temp[:,1])
    obj_label_list = [int(float(i)) for i in obj_label_list]
    return image_list, obj_label_list

#%%
#因為讀數的image和label都是feature, label轉乘int64 image轉成bytes數據格式
def feature_int64(value):
    if not isinstance(value, list):#判斷value是否為list格式
        value = [value]
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def feature_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%

#開始轉換 data轉成TFRecord, 把image 和label都轉到一個TFrecord file
def convert_to_tfrecord(images, labels, save_dir, name):

    filename = os.path.join(save_dir, name + '.tfrecords')
    n_samples = len(labels)
    
    if np.shape(images)[0] != n_samples:
        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, n_samples):
        try:
            image = Image.open(images[i]) # type(image) must be array!
            image = image.resize((227,227)) 
            image_raw = image.tobytes()
            label = int(labels[i])
            example = tf.train.Example(features = tf.train.Features(feature = {
                                        'label':feature_int64(label),
                                        'image_raw': feature_bytes(image_raw)}))
            writer.write(example.SerializeToString())
           
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')
    
#%%  
#read data from tfrecord
def read_and_decode( tfrecord_file, batch_size, img_W, img_H, n_classes ):
    filename_queue = tf.train.string_input_producer( [tfrecord_file])#生成一個對列queue
    #宣告一個讀取
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    #開始解析資料
    img_feature = tf.parse_single_example( serialized_example,
                                        features = {
                                                  'label': tf.FixedLenFeature([],tf.int64),
                                                  'image_raw': tf.FixedLenFeature([], tf.string)})
    #解析完然後decode 一開始的格式
    image  = tf.decode_raw(img_feature['image_raw'], tf.uint8 )

    image = tf.reshape( image, [img_W, img_H,3] )
    #image = tf.image.per_image_standardization(image)
    label = tf.cast(img_feature['label'], tf.int32)
    '''image_batch, label_batch = tf.train.batch( [image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = 2000)'''


    image_batch, label_batch = tf.train.shuffle_batch( [image, label],
                                              batch_size = batch_size,
                                              num_threads = 64,
                                              capacity = 2000,
                                              min_after_dequeue=1000 )                                        

    image_batch = tf.cast(image_batch, tf.float32) #一次batch的張量
    #one-hot
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    
    print('label_batch:', label_batch)
    print('image_batch:', image_batch)
    return image_batch, label_batch
    
#%%
