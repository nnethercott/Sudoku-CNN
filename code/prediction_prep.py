from skimage.transform import resize
from skimage.util import crop

import segmentation as s
import dataprep as dp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import prediction_prep as pp


def segment_predict(image, size):  #use min of heightstep and widthstep -1
    image = s.imprep(image, size)
    image = np.reshape(image, [size,size])
    for i in range((int)(size/6)):
        for j in range(size):
            image[i,j]=0.985
            image[j,i]=0.985
            image[size-1-(int)(i/2),j]=0.985
            image[j,size-1-(int)(i/2)]=0.985

    for i in range(size):
        for j in range(size):
            if image[i,j]>0.90:
                image[i,j]=1.0

    #detecting empty squares
    sum = 0

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            sum += image[i,j]

    ave = sum/(image.size)

    #centering the image on assumption of lower right positioning
    '''image = np.reshape(image, [size,size,1])
    image = crop(image, (((int)(size/10), (int)(size/30)), ((int)(size/9), (int)(size/30)), (0,0)))
    image = np.reshape(image, [image.shape[0],image.shape[1]])'''

    #making image less intense
    '''for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]>0.40:
                image[i,j]=1.0'''

    #recognizing numbers or lack thereof
    if ave<0.99:
        image = resize(image, (32, 32), anti_aliasing=True)
        image = np.reshape(image, [1,32,32,1])
        '''model1 = tf.keras.models.load_model("SNIST4.0")
        prediction1 = model1.predict(image)'''
        model2 = tf.keras.models.load_model("SNIST6.1")
        prediction2 = model2.predict(image)
        predictions = prediction2
        prediction = np.argmax(predictions)+1   #since model trains with first entry as 1


    else:
        prediction = 0

    return prediction
