import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#Import data 
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

#Check data size
# print(x_train.shape) #60000,28,28
# print(y_train.shape) #60000,
# print(x_test.shape)  #10000,28,28
# print(y_test.shape)  #10000

#Print out the data image
# print(y_train[6])
# plt.imshow(x_train[6], cmap='gray')
# plt.show()

#Reshape the array to 4-dims,so that it can work with Keras's API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Change the value to float type for the normalization division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalization
x_train = x_train / 255
x_test  = x_test  / 255

print(x_train.shape[0])
