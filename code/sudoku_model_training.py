'''could look into modules for opening file data, but rather just get to work
on the actual backbone, i.e. the model.  So, manually crop 27 images of numbers from
sudoku puzzle image, then use the data_extend function to create new arrays of training
data '''

import segmentation as s
import dataprep as dp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize, rotate, swirl, rescale
from skimage.util import crop, random_noise
import SNIST as sn


images = []
'''instead of this i could hardcode the images in there, but this seems like
 good foundations for implementing file reading in the future'''


#IMAGES ARE TOO ZOOMED IN COMPARED TO HOW THEY ARE PRESENTED IN PUZZLE
 #loading in all the image and variants

image1 = Image.open('1.13.jpg')
image2 = Image.open('1.23.jpg')
image3 = Image.open('1.33.jpg')
image4 = Image.open('2.13.jpg')
image5 = Image.open('2.23.jpg')
image6 = Image.open('2.33.jpg')
image7 = Image.open('3.13.jpg')
image8 = Image.open('3.23.jpg')
image9 = Image.open('3.33.jpg')
image10 = Image.open('4.13.jpg')
image11 = Image.open('4.23.jpg')
image12 = Image.open('4.33.jpg')
image13 = Image.open('5.13.jpg')
image14 = Image.open('5.23.jpg')
image15 = Image.open('5.33.jpg')
image16 = Image.open('6.13.jpg')
image17 = Image.open('6.23.jpg')
image18 = Image.open('6.33.jpg')
image19 = Image.open('7.13.jpg')
image20 = Image.open('7.23.jpg')
image21 = Image.open('7.33.jpg')
image22 = Image.open('8.13.jpg')
image23 = Image.open('8.23.jpg')
image24 = Image.open('8.33.jpg')
image25 = Image.open('9.13.jpg')
image26 = Image.open('9.23.jpg')
image27 = Image.open('9.33.jpg')

image28 = Image.open('1.12.jpg')
image29 = Image.open('1.22.jpg')
image30 = Image.open('1.32.jpg')
image31 = Image.open('2.12.jpg')
image32 = Image.open('2.22.jpg')
image33 = Image.open('2.32.jpg')
image34 = Image.open('3.12.jpg')
image35 = Image.open('3.22.jpg')
image36 = Image.open('3.32.jpg')
image37 = Image.open('4.12.jpg')
image38 = Image.open('4.22.jpg')
image39 = Image.open('4.32.jpg')
image40 = Image.open('5.12.jpg')
image41 = Image.open('5.22.jpg')
image42 = Image.open('5.32.jpg')
image43 = Image.open('6.12.jpg')
image44 = Image.open('6.22.jpg')
image45 = Image.open('6.32.jpg')
image46 = Image.open('7.12.jpg')
image47 = Image.open('7.22.jpg')
image48 = Image.open('7.32.jpg')
image49 = Image.open('8.12.jpg')
image50 = Image.open('8.22.jpg')
image51 = Image.open('8.32.jpg')
image52 = Image.open('9.12.jpg')
image53 = Image.open('9.22.jpg')
image54 = Image.open('9.32.jpg')

image55 = Image.open('Screenshot (90).jpg')
image56 = Image.open('Screenshot (91).jpg')
image57 = Image.open('Screenshot (92).jpg')
image58 = Image.open('Screenshot (93).jpg')
image59 = Image.open('Screenshot (94).jpg')
image60 = Image.open('Screenshot (95).jpg')
image61 = Image.open('Screenshot (96).jpg')
image62 = Image.open('Screenshot (97).jpg')
image63 = Image.open('Screenshot (98).jpg')
image64 = Image.open('Screenshot (99).jpg')
image65 = Image.open('Screenshot (100).jpg')
image66 = Image.open('Screenshot (101).jpg')
image67 = Image.open('Screenshot (102).jpg')
image68 = Image.open('Screenshot (103).jpg')
image69 = Image.open('Screenshot (104).jpg')
image70 = Image.open('Screenshot (105).jpg')
image71 = Image.open('Screenshot (106).jpg')
image72 = Image.open('Screenshot (107).jpg')
image73 = Image.open('Screenshot (108).jpg')
image74 = Image.open('Screenshot (109).jpg')
image75 = Image.open('Screenshot (110).jpg')
image76 = Image.open('Screenshot (111).jpg')
image77 = Image.open('Screenshot (112).jpg')
image78 = Image.open('Screenshot (113).jpg')
image79 = Image.open('Screenshot (114).jpg')
image80 = Image.open('Screenshot (115).jpg')
image81 = Image.open('Screenshot (116).jpg')
image82 = Image.open('Screenshot (117).jpg')
image83 = Image.open('Screenshot (118).jpg')
image84 = Image.open('Screenshot (119).jpg')
image85 = Image.open('Screenshot (120).jpg')
image86 = Image.open('Screenshot (121).jpg')
image87 = Image.open('Screenshot (122).jpg')
image88 = Image.open('Screenshot (123).jpg')
image89 = Image.open('Screenshot (124).jpg')
image90 = Image.open('Screenshot (125).jpg')
image91 = Image.open('Screenshot (126).jpg')
image92 = Image.open('Screenshot (127).jpg')
image93 = Image.open('Screenshot (128).jpg')
image94 = Image.open('Screenshot (129).jpg')
image95 = Image.open('Screenshot (130).jpg')
image96 = Image.open('Screenshot (131).jpg')
image97 = Image.open('Screenshot (132).jpg')
image98 = Image.open('Screenshot (133).jpg')
image99 = Image.open('Screenshot (134).jpg')

image100 = Image.open('1.4.jpg')
image101 = Image.open('2.4.jpg')
image102 = Image.open('3.4.jpg')
image103 = Image.open('4.4.jpg')
image104 = Image.open('5.4.jpg')
image105 = Image.open('6.4.jpg')
image106 = Image.open('7.4.jpg')
image107 = Image.open('8.4.jpg')
image108 = Image.open('9.4.jpg')

image109 = Image.open('1.1.jpg')
image110 = Image.open('1.2.jpg')
image111 = Image.open('1.3.jpg')
image112 = Image.open('2.1.jpg')
image113 = Image.open('2.2.jpg')
image114 = Image.open('2.3.jpg')
image115 = Image.open('3.1.jpg')
image116 = Image.open('3.2.jpg')
image117 = Image.open('3.3.jpg')
image118 = Image.open('4.1.jpg')
image119 = Image.open('4.2.jpg')
image120 = Image.open('4.3.jpg')
image121 = Image.open('5.1.jpg')
image122 = Image.open('5.2.jpg')
image123 = Image.open('5.3.jpg')
image124 = Image.open('6.1.jpg')
image125 = Image.open('6.2.jpg')
image126 = Image.open('6.3.jpg')
image127 = Image.open('7.1.jpg')
image128 = Image.open('7.2.jpg')
image129 = Image.open('7.3.jpg')
image130 = Image.open('8.1.jpg')
image131 = Image.open('8.2.jpg')
image132 = Image.open('8.3.jpg')
image133 = Image.open('9.1.jpg')
image134 = Image.open('9.2.jpg')
image135 = Image.open('9.3.jpg')



x_train = sn.extend385_2(image1)
images.append(image2)
images.append(image3)
images.append(image4)
images.append(image5)
images.append(image6)
images.append(image7)
images.append(image8)
images.append(image9)
images.append(image10)
images.append(image11)
images.append(image12)
images.append(image13)
images.append(image14)
images.append(image15)
images.append(image16)
images.append(image17)
images.append(image18)
images.append(image19)
images.append(image20)
images.append(image21)
images.append(image22)
images.append(image23)
images.append(image24)
images.append(image25)
images.append(image26)
images.append(image27)
images.append(image28)
images.append(image29)
images.append(image30)
images.append(image31)
images.append(image32)
images.append(image33)
images.append(image34)
images.append(image35)
images.append(image36)
images.append(image37)
images.append(image38)
images.append(image39)
images.append(image40)
images.append(image41)
images.append(image42)
images.append(image43)
images.append(image44)
images.append(image45)
images.append(image46)
images.append(image47)
images.append(image48)
images.append(image49)
images.append(image50)
images.append(image51)
images.append(image52)
images.append(image53)
images.append(image54)
images.append(image55)
images.append(image56)
images.append(image57)
images.append(image58)
images.append(image59)
images.append(image60)
images.append(image61)
images.append(image62)
images.append(image63)
images.append(image64)
images.append(image65)
images.append(image66)
images.append(image67)
images.append(image68)
images.append(image69)
images.append(image70)
images.append(image71)
images.append(image72)
images.append(image73)
images.append(image74)
images.append(image75)
images.append(image76)
images.append(image77)
images.append(image78)
images.append(image79)
images.append(image80)
images.append(image81)
images.append(image82)
images.append(image83)
images.append(image84)
images.append(image85)
images.append(image86)
images.append(image87)
images.append(image88)
images.append(image89)
images.append(image90)
images.append(image91)
images.append(image92)
images.append(image93)
images.append(image94)
images.append(image95)
images.append(image96)
images.append(image97)
images.append(image98)
images.append(image99)
images.append(image100)
images.append(image101)
images.append(image102)
images.append(image103)
images.append(image104)
images.append(image105)
images.append(image106)
images.append(image107)
images.append(image108)
images.append(image109)
images.append(image110)
images.append(image111)
images.append(image112)
images.append(image113)
images.append(image114)
images.append(image115)
images.append(image116)
images.append(image117)
images.append(image118)
images.append(image119)
images.append(image120)
images.append(image121)
images.append(image122)
images.append(image123)
images.append(image124)
images.append(image125)
images.append(image126)
images.append(image127)
images.append(image128)
images.append(image129)
images.append(image130)
images.append(image131)
images.append(image132)
images.append(image133)
images.append(image134)
images.append(image135)




#combining training images into larger array
for i in range(len(images)):
    #np.concatenate allows you to append multidimensional arrays assuming same dimensionality
    x_train = np.concatenate((x_train, sn.extend385_2(images[i])), axis=0)

print(x_train.shape)

y_train = []
for k in range(2):
    for i in range(9):
        for j in range(1155):
            y_train.append(i)

for i in range(9):
    for j in range(1925):
        y_train.append(i)


for i in range(9):
    for j in range(385):
        y_train.append(i)

for i in range(9):
    for j in range(1155):
        y_train.append(i)


y_train = np.array(y_train)
y_train = np.reshape(y_train, [51975,1])

x_train = np.reshape(x_train, [51975,36,36,1])  #have no clue as to why you need this for the convnet to work

'''#get back to this and just run through every single picture to verify no outliers
y = sn.extend715(image54)
plt.imshow(y[0])
plt.show()'''


#shuffle data to prevent skewed learning
import random
combined = list(zip(x_train, y_train))
random.shuffle(combined)
x_train, y_train = zip(*combined)

#reformat all of these datasets cause I think it thinks these are no longer np arrays
x_train = np.array(x_train)
x_train = np.reshape(x_train, [51975, 36,36,1])
y_train = np.array(y_train)
y_train = np.reshape(y_train, [51975, 1])

#validate shuffle accuracy
'''for i in range(20):
    plt.imshow(x_train[i])
    plt.show()
    print(y_train[i])'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D

model = tf.keras.models.Sequential()
model.add(Conv2D(64, (3,3), input_shape=(x_train.shape[1:]), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=1))
model.add(Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.50))
model.add(tf.keras.layers.Dense(9, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=100)


model.save('SNIST6.3') #hahaha
