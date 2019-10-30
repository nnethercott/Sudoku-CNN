import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def loadimage(filename):
    image = Image.open(filename)
    return image

def imresize(image, i, j): #ask barrett if we can have tuple as argument
    resized = image.resize((i,j))
    return resized

def zeromatrix(image):
    image_arr=[]
    for m in range(image.size[0]):
        image_arr.append([])
    for m in range(image.size[0]):
        for n in range(image.size[1]):
            image_arr[m].append(0)
    return image_arr

def loadvalues(image, temp):
    for m in range(image.size[0]):
        for n in range(image.size[1]):
            temp[n][m]=image.getpixel((m,n))

    temp=np.array(temp)
    temp = temp.astype('float32')
    temp /= 255
    return temp

def reshape(image):
        image = np.reshape(image, [1,image.size[0],image.size[1],image.getpixel[0][0]]) #might get rid of this if appending a list
        return image
