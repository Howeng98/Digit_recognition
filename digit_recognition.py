import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
# Normalize
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1), \
x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1)

# Debug 用的資訊
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 參數
batch_size = 32
num_classes = 10
epochs = 5

# 建立 model
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
activation='relu',
input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

# 畫出 model ，圖放在後面
keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
# 訓練
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
# 驗證
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)