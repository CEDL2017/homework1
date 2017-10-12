# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:58:32 2017

@author: hsuan
"""


import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import numpy as np
import re 

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='train')
parser.add_argument('--test_dir', default='test')
parser.add_argument('--labels_path', default= 'labels' , type=str)
parser.add_argument('--model_path', default='vgg_16.ckpt', type=str)   #
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=1, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]


def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    places = os.listdir(directory)
    #frames = []
    framesnames = []

    for place in places:
        for num in os.listdir( os.path.join(directory, place) ):
            #for h in os.listdir( os.path.join(directory, place, num) ):
            frames = []    
            for f in os.listdir( os.path.join(directory, place, num, 'Lhand') ):
                    #print( f )
                frames.append( f )
            frames = sorted(frames, key=lambda x: int(re.sub('\D', '', x)))
            for f in frames:
                framesnames.append( os.path.join(directory, place, num, 'Lhand', f) )
                
            frames = []
            for f in os.listdir( os.path.join(directory, place, num, 'Rhand') ):
                    #print( f )
                frames.append( f )
            frames = sorted(frames, key=lambda x: int(re.sub('\D', '', x)))
            for f in frames:
                framesnames.append( os.path.join(directory, place, num, 'Rhand', f) )
        
    
    framesnames = list(framesnames)
    #for f in os.listdir( directory ):
            #framesnames.append(os.path.join(directory) )
    
   
    print( framesnames )
        
    return framesnames


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    test_filenames = list_images(args.test_dir)
    labels_path = args.labels_path
    
    test_labels = []
    for i in range(4, -1):
        loadf1 = np.load( os.path.join(labels_path, 'house', 'obj_left'+str(i)+'.npy') ) 
        {test_labels.append(l) for l in loadf1.astype(int) }
        loadf2 = np.load( os.path.join(labels_path, 'house', 'obj_right'+str(i)+'.npy') )
        {test_labels.append(l) for l in loadf2.astype(int) }
    for i in range(5, -1):
        loadf1 = np.load( os.path.join(labels_path, 'lab', 'obj_left'+str(i)+'.npy') ) 
        {test_labels.append(l) for l in loadf1.astype(int) }
        loadf2 = np.load( os.path.join(labels_path, 'lab', 'obj_right'+str(i)+'.npy') )
        {test_labels.append(l) for l in loadf2.astype(int) }
    for i in range(4, -1):
        loadf1 = np.load( os.path.join(labels_path, 'office', 'obj_left'+str(i)+'.npy') ) 
        {test_labels.append(l) for l in loadf1.astype(int) }
        loadf2 = np.load( os.path.join(labels_path, 'office', 'obj_right'+str(i)+'.npy') )
        {test_labels.append(l) for l in loadf2.astype(int) }
        
    print(test_labels)

    num_classes = 24

 
    graph = tf.Graph()
    with graph.as_default():
      
        def _parse_function(filename, label):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_png(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            smallest_side = 256.0
            height, width = tf.shape(image)[0], tf.shape(image)[1]
            height = tf.to_float(height)
            width = tf.to_float(width)

            scale = tf.cond(tf.greater(height, width),
                            lambda: smallest_side / width,
                            lambda: smallest_side / height)
            new_height = tf.to_int32(height * scale)
            new_width = tf.to_int32(width * scale)

            resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
            return resized_image, label
        def training_preprocess(image, label):
            crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
            flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = flip_image - means                                     # (5)

            return centered_image, label


   
        # Training dataset
        test_filenames = tf.constant(test_filenames)
        test_labels = tf.constant(test_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((test_filenames, test_labels))
        train_dataset = train_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.map(training_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)

       


  
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        #val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

 
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        # Specify where the model checkpoint is (pretrained weights).
        model_path = args.model_path
        assert(os.path.isfile(model_path))

 
        variables_to_restore = tf.contrib.framework.get_variables_to_restore('vgg_16')
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

       


        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate2)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy)
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

        saver.restore(sess, "tmp/model")
        print("restore model !!!!")
   
    with tf.Session(graph=graph) as sess:
        init_fn(sess)  # load the pretrained weights
        #sess.run(fc8_init)  # initialize the new fc8 layer

       


        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs2):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            #val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            #print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)