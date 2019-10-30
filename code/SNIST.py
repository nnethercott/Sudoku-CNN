import segmentation as s
import dataprep as dp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize, rotate, swirl, rescale
from skimage.util import crop, random_noise

def data_extend(image, angle, crop_num, prelim_size, final_size):
    array_out = []
    #import as 100x100 numpy array, perform distortions and resize to 32x32 to train
    image = s.imprep(image, prelim_size)
    image = np.reshape(image, [prelim_size, prelim_size])    #consider changing the imprep function because its so stupid

    angles = np.linspace(-angle, angle, 2*angle+1)
    for a in angles:
        a = int(a)
        image2 = rotate(image, a)
        for i in range(abs(a)):     #gets rid of black border from rotation, f(angle)
            for j in range(prelim_size):
                image2[i,j]=0.985
                image2[j,i]=0.985
                image2[prelim_size-1-i,j]=0.985
                image2[j,prelim_size-1-i]=0.985

        gradient = np.linspace(0.90, 1, 11)

        for colour in gradient:
            for i in range(prelim_size):
                for j in range(prelim_size):
                    if image2[i,j]>0.5:
                        image2[i,j]=colour


            for c in range(crop_num):
                image3 = crop(image2, c)
                image3 = resize(image3, (final_size, final_size))
                array_out.append(image3)

    array_out = np.array(array_out)
    return array_out

def extend715(image):
    array = data_extend(image, 6, 5, 100, 32)
    return array

def extend385(image):
    array = data_extend(image, 3, 5, 100, 32)
    return array

def extend385_2(image):
    array = data_extend(image, 3, 5, 100, 36)
    return array

'''code is good so far, but not too many variations with current structure
consider adding maybe a swirl and reducing the number of outputs and then training
with 3 distinct pictures possibly.  could also add some gaussian noise'''
