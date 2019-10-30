import dataprep as dp

def imcrop(image, i, j, m, n): #ask barrett if we can have tuple as argument
    cropped = image.crop((i,j,m,n))
    return cropped

#converts image to numpy array of pixel data
def numpify(image):      #might want to take zero matrix outta this so you save memory
    zero = dp.zeromatrix(image)
    nump = dp.loadvalues(image, zero)
    return nump

def c2g(rgb):
    r, g, b = rgb[:,:,:,0], rgb[:,:,:,1], rgb[:,:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def imprep(image, size):  #takes numpy pixels and reshapes and converts to grayscale for MNIST
    image = dp.imresize(image, size, size)
    image = numpify(image)
    image = np.reshape(image, [1,size, size, 3])
    image = c2g(image)
    image = np.reshape(image, [1,size,size,1])
    return image


import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
image = Image.open('Image-12.jpg')
image = imprep(image, 28)

new_model = tf.keras.models.load_model("MNIST")

predictions = new_model.predict(image)
print(predictions)
print("this is a {}". format(np.argmax(predictions)))'''

'''
#if one pixel and opposite are within a threshold ratio of one another turn them white
#maybe problem if threshold too low, also if grid skews between pixel columns
#could make a function that just further crops the image like 15% or something
def nogrid(image, size, threshold):
    sum_h = np.array(np.zeros(size))
    ave_h = np.array(np.zeros(size))
    sum_v = np.array(np.zeros(size))
    ave_v = np.array(np.zeros(size))

    for i in range(size):
        for j in range(size):
            sum_h[i] = sum_h[i] + image[i,j]
            ave_h[i] = sum_h[i]/size

            sum_v[i] = sum_v[i] + image[j,i]
            ave_v[i] = sum_v[i]/size

    h_correction = []
    v_correction = []
    for k in range(size):
        if ave_h[k]>threshold:
            h_correction.append(k)

        if ave_v[k]>threshold:
            v_correction.append(k)

    for i in range(len(h_correction)):
        for j in range(size):
            image[h_correction[i],j] = 0

    for i in range(len(v_correction)):
        for j in range(size):
            image[j,v_correction[i]] = 0

    return image

#other ideas include model a blur like a normal distribution from center radially outwards
#the cropping code should capture the number but likely some borders so pixel intensity
#could be scaled by 1/normal[i]

image = Image.open('Image-20.jpg')
image = imprep(image, 28)

image = nogrid(image, 28, 0.7)
print(image)'''
