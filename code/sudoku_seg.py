import segmentation as s
import dataprep as dp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import prediction_prep as pp



def prediction_generator(image_original, num): #Breaks images into multiple equal sized images
    list = []
    width, height = image_original.size
    wStep =(int)(width/num)
    hStep=(int)(height/num)
    cropsize = min(wStep, hStep) - 1

    for i in range(0,height,hStep):
        if (i + hStep > height):
            break
        for j in range(0,width,wStep):
            if (j + wStep > width):
                break
            imgCropped = image_original.crop((j,i,j+wStep,i+hStep))
            #now with imgCropped implement code for predictions
            list.append(pp.segment_predict(imgCropped, cropsize))
    board = np.reshape(list, [num,num])
    print(board)
    return board
