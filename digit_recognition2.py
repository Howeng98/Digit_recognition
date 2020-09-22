'''
21/09/2020
practice for project  - digit_recognition

'''
import gzip
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

x_train = gzip.open('mnist/train-images-idx3-ubyte.gz', 'rb')
y_train = gzip.open('mnist/train-labels-idx1-ubyte.gz', 'rb')
x_test = gzip.open('mnist/t10k-images-idx3-ubyte.gz', 'rb')
y_test = gzip.open('mnist/t10k-labels-idx1-ubyte.gz', 'rb')

x_train.read(16)
y_train.read(16)
x_test.read(16)
y_test.read(16)

image_size = 28
num_images = 50

x_train = x_train.read(image_size*image_size*num_images)
x_train = np.frombuffer(x_train, dtype=np.uint8).astype(np.float32)
x_train = x_train.reshape(num_images, image_size, image_size, 1)

x_test = x_test.read(image_size*image_size*num_images)
x_test = np.frombuffer(x_test, dtype=np.uint8).astype(np.float32)
x_test = x_test.reshape(num_images, image_size, image_size, 1)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_train = x_train / 255
x_test  = x_test / 255


batch_size = 20
num_classes = 10
epochs = 10

# setup model
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu',input_shape=(image_size,image_size,1)))
model.add(keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accurancy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

test_loss, test_acc = model.avaluate(x_test,y_test)
print('Test accurancy:', test_acc)