import tensorflow as tf
import numpy as np
from PIL import Image
#%%

def get_image_Files( file_dir, image_number ) : #回傳影像list的

    hand=[]
    for k in range(1, image_number+1, +1):
        path = file_dir+"Image"+str(k)+".png"
        hand.append(path)
        
    #print('There are %d hand' %(len(hand)))
    return hand

#%%
def get_labe_Files(lab_dir): #回傳label的list，和數量
    label_hand = []
    label_hand = np.load(lab_dir)
    #print('There are %d label' %(len(label_hand)))
    
    label_hand = [int(i) for i in label_hand]
    
    image_numbr = len(label_hand)
    
    return image_numbr, label_hand
    

#%%
