import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from load_datas import *

label_path='../labels'

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='../frames/train')
parser.add_argument('--val_dir', default='../frames/test')
parser.add_argument('--model_path', default='./saved_model/model.ckpt.data-00000-of-00001', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=15, type=int)
parser.add_argument('--num_epochs2', default=5, type=int)
parser.add_argument('--learning_rate1', default=1e-3, type=float)
parser.add_argument('--learning_rate2', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

VGG_MEAN = [123.68, 116.78, 103.94]

def list_images(directory):
    scenes = os.listdir(directory)
    filenames = []
    for scene in scenes:
        scene_path = os.path.join(directory,scene)
        videos = os.listdir(scene_path)
        for video in videos:
            video_path = os.path.join(scene_path, video)
            left_path = os.path.join(video_path, 'Lhand')
            fs = os.listdir(left_path)
            for f in fs:
                file_path = os.path.join(left_path, f)
                filenames.append(file_path)
            right_path = os.path.join(video_path, 'Rhand')
            fs = os.listdir(right_path)
            for f in fs:
                file_path = os.path.join(right_path, f)
                filenames.append(file_path)
    filenames = list(filenames)
    return filenames


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

def list_flatten(label):
    label_f = []
    for i in range(len(label)):
        temp = label[i].tolist()
        label_f = label_f + temp
    return(label_f)
    
def fit_batch(data,label,batch_size):
    rm = len(data) % batch_size 
    if(rm!=0):    
        data = data[:-rm]
        label = label[:-rm]
    return(data,label)

def label_to_int(labels):
    unique_labels = list(set(labels))
    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i
    labels = [label_to_int[l] for l in labels]
    return labels

def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_filenames = list_images(args.train_dir)
    val_filenames = list_images(args.val_dir)
    train_labels, val_labels = load_all_labels(label_path, 'obj',0)
    train_labels =  list_flatten(train_labels)
    val_labels =    list_flatten(val_labels)
    train_labels = label_to_int(train_labels)
    val_labels = label_to_int(val_labels)        
  
    train_filenames,train_labels = fit_batch(train_filenames,train_labels,args.batch_size)
    val_filenames, val_labels = fit_batch(val_filenames,val_labels,args.batch_size)
    num_classes = len(set(train_labels))
 
    graph = tf.Graph()
    with graph.as_default():
        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image so its smaller side is 256 pixels long
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
        # Preprocessing (for validation)
        # (3) Take a central 224x224 crop to the scaled image
        # (4) Substract the per color mean `VGG_MEAN`
        # Note: we don't normalize the data here, as VGG was trained without normalization
        def val_preprocess(image, label):
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means                                     # (4)

            return centered_image, label

        # Validation dataset
        
        val_filenames = tf.constant(val_filenames)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_filenames, val_labels))
        val_dataset = val_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        val_dataset = val_dataset.map(val_preprocess,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)

        iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types,
                                                           batched_val_dataset.output_shapes)
        images, labels = iterator.get_next()
        val_init_op = iterator.make_initializer(batched_val_dataset)
        is_training = tf.placeholder(tf.bool)
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(images, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        model_path = args.model_path
        assert(os.path.isfile(model_path))
	
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)

        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        tf.get_default_graph().finalize()

    with tf.Session(graph=graph) as sess:
        checkpoint_path = './saved_model'
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir = checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
        val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)   
        print('Val accuracy: %f\n' % val_acc)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    