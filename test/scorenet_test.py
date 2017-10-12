import _init_paths
from scorenet import ScoreNet
from keras.utils import np_utils
from config import *
import numpy as np
import coder
import time
import cv2
import threading
import tensorflow as tf 

# img_dir = '../img/scorenet/video6_frame090.bmp'
# Tensorflow graph object
graph = tf.get_default_graph()
# Camera capture related variable
cap = None
is_cap_open = False

# model object
scoring_model = None

# break variable and timer to calculate fps
frame_index = 0
timer = 0.0
fps_index = 0

# ---------------------------------------------------------------
# Image objects and predict result
# ---------------------------------------------------------------
# 1st
frame_fetch = None

# 2nd
frame_process = None
result_scoring = None

# 3rd
show_frame = None
scoring_input = None

# scoring_true = np_utils.to_categorical(np.load('../label.npy'),5)

class fetchImgThread(threading.Thread):
    """
        The thread to fetch the image in each duration
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def start(self):
        threading.Thread.__init__(self)
        threading.Thread.start(self)
    
    def join(self, _sec):
        threading.Thread.join(self, _sec)

    def run(self):
        global cap
        global is_cap_open
        global frame_fetch
        _, frame = cap.read()
        is_cap_open = _
        frame_fetch = cv2.resize(frame, (480, 270))

class deepThread(threading.Thread):
    """
        The thread to do the segmentation and classification
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def start(self):
        threading.Thread.__init__(self)
        threading.Thread.start(self)

    def join(self, _sec):
        threading.Thread.join(self, _sec)

    def run(self):
        global segment_model
        global frame_process
        global result_segment
        global result_scoring
        global graph
        global sess
        global frame_index

        with graph.as_default():
            # x_test = np.expand_dims(frame_process, axis=0)
            if frame_index % 3 == 0 and frame_process.any():
                result_scoring = scoring_model.test(
                    np.expand_dims(frame_process, 0)
                )      
        
# # Test
if __name__ == '__main__':
    
    scoring_model = ScoreNet(save_path='../model/scorenet.h5')
    # scoring_model.compile()
    # cap = cv2.VideoCapture('../video/1.mp4')
    cap = cv2.VideoCapture(video_name)    

    fetch_thread = fetchImgThread()
    deep_thread = deepThread()
    
    fetch_thread.start()
    fetch_thread.join(1)
    # Pass the input object and clean the previous status
    frame_process = np.copy(frame_fetch)
    frame_fetch = None
    
    # Grab 2nd frame
    fetch_thread.start()
    deep_thread.start()
    fetch_thread.join(5)
    deep_thread.join(5)
    
    show_frame = np.copy(frame_process)
    scoring_input = np.copy(result_scoring)
    result_scoring = None
    frame_process = np.copy(frame_fetch)
    frame_fetch = None
    
    while cap.isOpened():
        _time = time.time()

        fetch_thread.start()
        deep_thread.start()
        fetch_thread.join(5)
        deep_thread.join(5)
        cv2.imshow('test',frame_process)
        cv2.imshow('scoring', coder.decodeByVector(show_frame, scoring_input))
        
        # Pring fps
        if timer > 1.0:
            print ("fps: ", fps_index / timer)
            fps_index = 0
            timer = 0

        # judge if we want to break
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
#         if not unicode(video_name).isnumeric():
#             if frame_index > break_frame_index:
#                 break

        # Pass the input object and clean the previous status
        show_frame = np.copy(frame_process)
        scoring_input = np.copy(result_scoring)
        # print('Scoring:',scoring_input)
        # loss,accuracy = scoring_model.evaluate(scoring_true,scoring_input)
        # print('test loss: ', loss)
        # print('test accuracy: ', accuracy)

        # Clear the scoring result if predict ScoreNet in next frame
        if frame_index % 3 == 2:
            result_scoring = None
        frame_process = np.copy(frame_fetch)
        frame_fetch = None

        # Update fps computation variable
        frame_index += 1
        fps_index += 1
        timer += (time.time() - _time)
        
    cap.release()
    
    # model = ScoreNet(save_path='../model/scorenet.h5')
    # x_test = np.expand_dims(cv2.imread(img_dir), axis=0)
    # _time = time.time()
    # prediction = model.test(x_test)
    # print ('time comsumption: ', time.time() - _time)
    # # Show the test result
    # prediction = prediction.astype(int)

    # res = coder.decodeByVector(x_test[0], prediction)
    # cv2.imshow('show', res)
    # cv2.waitKey(0)

