# Sudoku-CNN

The codes included were used primarily for the purposes of data prep and image processing since keras
requires numpy arrays for operation. Additionally, a sudoku solver .py is included along with the code for
training the SNIST models.

In order to approach the issue of solving a sudoku puzzle from an image, a 2D convolutional neural network
trained to identify digits in typed font (primarily arial) was developed.  The overall code functions by
segmenting the image into 81 squares, performing basic manipulations to get rid of borders and to increase
constrast of the digit and the background, using the aforementioned neural network to make a prediction as to
what the digit is, and then storing it in a list which is later formatted into a 9 by 9 matrix.  
