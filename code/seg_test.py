'''border detection and correction is outside the scope right now since I don't
want to look into it too much other than what i've done so far.  The goal is to
make a code that solves sudoku puzzles and you can sit around and mess with the
little features and tweak it over and over again to make it more versatile.  For
this reason, the project scope is to assume the sudoku picture is given pre-cropped
and square.  We will then just clean the image a bit with some piece-wise functions
and then crop the image x amount to assure no border. '''


import segmentation as s
import dataprep as dp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#to be used after barrett segments the image into 81 images
def findline(image, size, threshold):
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

            list1 = []
            list2 = []
    for i in range(size):
        if ave_h[i]<threshold:   #ad hoc
            list1.append(i)
        if ave_v[i]<threshold:
            list2.append(i)

    for i in range(len(list1)):
        for j in range(size):
            image[list1[i],j]=1.0

    for i in range(len(list2)):
        for j in range(size):
            image[j,list2[i]]=1.0

    return image

#note: changing prelim size changes prediction confidence despite inevitable downsize to 28x28
'''image = findline(image, 100, 0.30)
image = np.reshape(image, [100,100])
plt.imshow(image)
plt.show()

from skimage.transform import resize
from skimage.util import crop

image = crop(image, 20)

image = resize(image, (32, 32), anti_aliasing=True)
image = np.reshape(image, [32,32])
plt.imshow(image)
plt.show()


image = np.reshape(image, [1,32,32,1])

new_model = tf.keras.models.load_model("SNIST")

predictions = new_model.predict(image)
print(predictions)
print("this is a {}". format(np.argmax(predictions)))
'''
