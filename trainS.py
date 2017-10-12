import os
import numpy as np
import tensorflow as tf
import math
import readfile
import input_data
import VGG19
#%%

N_CLASSES = 24
IMG_W = 227  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 227
BATCH_SIZE = 16
CAPACITY = 20000
MAX_STEP = 30000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.01 # with current parameters, it is suggested to use learning rate<0.0001
EPOCH = 50

#%%
def run_training():
    train_log_dir = '/home/viplab/Desktop/vgg_var2/log/train/'  
    tfrecords_file ='/home/viplab/Desktop/vgg_var2/tfrecord/train.tfrecords'
    save_dir = '/home/viplab/Desktop/vgg_var2/tfrecord/'
    image_list, obj_label_list = readfile.read_file()
    N_SAMPLES = len(obj_label_list)

    readfile.convert_to_tfrecord(image_list, obj_label_list, save_dir, 'train')

    train_batch, label_batch = readfile.read_and_decode(tfrecords_file, BATCH_SIZE, IMG_W, IMG_H, N_CLASSES ) 
    print('train_batch',train_batch)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, N_CLASSES])
    
    train_logits = VGG19.VGG19(x,  N_CLASSES)
    train_loss = VGG19.losses(train_logits, y)        
    train__acc = VGG19.evaACC(train_logits, y)
    train_op = VGG19.trainning(train_loss, learning_rate)   
    #將之前定義的所有summary op整合在一起
    summary_op = tf.summary.merge_all()
    
    
    saver = tf.train.Saver() #存module
    sess = tf.Session()

    #對所有變量初始化
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #創建一個file writer 用來寫數據
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    counter = 0
    k = 0
    try:
        for step in np.arange(0,EPOCH):
            print("epoch: ", step)
            k = 0
            while k< int(N_SAMPLES/BATCH_SIZE):
                if coord.should_stop():
                    break
                #print('train_batch',train_batch)
                image, label = sess.run([train_batch, label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc], feed_dict={x:image,y: label} )            
                counter +=1
                print('Step %d, train loss = %.2f, train accuracy = %.4f%%' %(counter, tra_loss, tra_acc*100))
                #if counter %50 = 0 :
                #summary_str = sess.run(summary_op)
                #train_writer.add_summary(summary_str, counter)
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=counter)
                k +=1

                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()



def evaluate():
    with tf.Graph().as_default():
        log_dir = '/home/viplab/Desktop/vgg_var2/log/train/' 
        tfrecords_file ='/home/viplab/Desktop/vgg_var2/tfrecord/test/test.tfrecords'
        save_dir = '/home/viplab/Desktop/vgg_var2/tfrecord/test/'        

        image_list, obj_label_list = readfile.read_test()
        N_SAMPLES = len(obj_label_list)

        readfile.convert_to_tfrecord(image_list, obj_label_list, save_dir, 'test')

        train_batch, label_batch = readfile.read_and_decode(tfrecords_file, BATCH_SIZE, IMG_W, IMG_H, N_CLASSES ) 
        print('train_batch',train_batch)


        train_logits = VGG19.VGG19(train_batch,  N_CLASSES)
        test__acc = VGG19.prediction_acc(train_logits, label_batch)
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("checkpoints processing...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                num_step = int(math.floor(N_SAMPLES / BATCH_SIZE))
                num_sample = num_step*BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(test__acc)
                    total_correct += np.sum(batch_correct)
                    step += 1
                    #readfile.plot_images(train_batch, label_batch, BATCH_SIZE)

                #print('Total testing samples: %d' %N_SAMPLES)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


choose = input('insert number : (1 train, 2 test) ')
if choose == '1' :
    run_training()
else:
    if choose == '2':
        evaluate()
    else:
        print('please insert 1 or 2:')
