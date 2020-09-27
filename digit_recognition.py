
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib
from random import randint
from matplotlib import pyplot
from tensorflow import keras
from mnist import MNIST
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model


# Check GPU device list and information
tf.test.gpu_device_name() 
device_lib.list_local_devices()
# !cat /proc/meminfo

# Loading train and test data
mndata = MNIST('drive/My Drive/Colab Notebooks/mnist')
x_train, y_train = mndata.load_training()
x_test, y_test   = mndata.load_testing()
x_train = np.array(x_train).astype('float32')/255.0
x_test = np.array(x_test).astype('float32')/255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape)
print(x_test.shape)

y_train = np_utils.to_categorical(y_train, 10)
y_test  = np_utils.to_categorical(y_test, 10)

# Variables
batch_size = 128
num_classes = 10
epochs = 10

# Setup Model
model = keras.Sequential()
model.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu',input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(32, kernel_size=(5,5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))
model.summary()

# Compile Model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test,y_test))

# Saving the Model
save_dir = 'drive/My Drive/Colab Notebooks'
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Load and evaluate
model = load_model("drive/My Drive/Colab Notebooks/keras_mnist.h5")
test_loss, test_acc = model.evaluate(x_test,np.array(y_test))
print('Test accuracy:', test_acc)

# Plotting the metrics
fig = plt.figure()
#plt.subplot(2,1,1)
plt.plot()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

#plt.subplot(2,1,2)
plt.plot()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Model Prediction
print(x_test.shape)
index = randint(1,9999)
plt.imshow(x_test[index].squeeze())
plt.show()

pred = model.predict_classes(x_test[index].reshape(1,28,28,1))
print("The predict output is:",pred)