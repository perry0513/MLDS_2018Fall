# build multi-layer perceptron (MLP) with Keras
import keras
from keras.utils import np_utils as kutil
from keras.models import Sequential
from keras.layers import Dense

import numpy as np

from matplotlib import pyplot as plt
from keras.datasets import mnist

# import MNIST data set
(x_train2D, y_train_label), (x_test2D, y_test_label) = mnist.load_data()

# convert to one-dimension
x_train = x_train2D.reshape(60000,784).astype('float32')
x_test = x_test2D.reshape(10000,784).astype('float32')

# normalization: mean=0, std=1
for i in range(len(x_train)):
    x=x_train[i]
    m=x.mean()
    s=x.std()
    x_train[i]=(x-m)/s
for i in range(len(x_test)):
    x=x_test[i]
    m=x.mean()
    s=x.std()
    x_test[i]=(x-m)/s

# shuffle y_train
np.random.shuffle(y_train_label)

# convert label to on-hot encoding
y_train = kutil.to_categorical(y_train_label)
y_test = kutil.to_categorical(y_test_label)

# MLP handwritten character recognition
model = Sequential()
model.add(Dense(input_dim=784,units=256,kernel_initializer='normal',activation='relu'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
#model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model fitting with training set
train = model.fit(x=x_train,y=y_train,validation_split=0.2,epochs=100,batch_size=100,verbose=2)

# display training history
plt.subplot(1,1,1)
plt.title('Shuffled Label')
plt.plot(train.history['loss'],label='train')
plt.plot(train.history['val_loss'],label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()