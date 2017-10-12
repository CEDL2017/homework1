import numpy as np
import cv2

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes = 24):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        
        self.read_class_list(class_list)
        
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,class_list):
        """
        Scan the image file and get the image paths and labels
        """
        with open(class_list) as f:
            lines = f.readlines()
            self.hand_images = []
            self.head_images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.hand_images.append(items[0])
                self.head_images.append(items[1])
                self.labels.append(int(items[2]))
            
            #store total number of data
            self.data_size = len(self.labels)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        # images = self.images.copy()
        # labels = self.labels.copy()
        hand_images = self.hand_images
        head_images = self.head_images
        labels = self.labels
        self.hand_images = []
        self.head_images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.hand_images.append(hand_images[i])
            self.head_images.append(head_images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        hand_paths = self.hand_images[self.pointer:self.pointer + batch_size]
        head_paths = self.head_images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images_hand = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        images_head = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        # print('len of hand path = ',len(hand_paths))
        # print('len of head path = ',len(head_paths))
        for i in range(len(hand_paths)):
            hand_img = cv2.imread(hand_paths[i])
            head_img = cv2.imread(head_paths[i])
            
            #flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                hand_img = cv2.flip(hand_img, 1)
            if self.horizontal_flip and np.random.random() < 0.5:
                head_img = cv2.flip(head_img, 1)
            
            #rescale image
            hand_img = cv2.resize(hand_img, (self.scale_size[0], self.scale_size[1]))
            hand_img = hand_img.astype(np.float32)
            head_img = cv2.resize(head_img, (self.scale_size[0], self.scale_size[1]))
            head_img = head_img.astype(np.float32)
            
            #subtract mean
            hand_img -= self.mean
            head_img -= self.mean
                   
            images_hand[i] = hand_img                                              
            images_head[i] = head_img

        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        # print(labels)
        # print('batch_size = ',batch_size)
        # print(one_hot_labels)
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images_hand, images_head, one_hot_labels
