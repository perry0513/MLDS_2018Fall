import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.callbacks import Callback
from matplotlib import pyplot as plt

def load_data(train_size):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[0:train_size]
	y_train = y_train[0:train_size]
	x_train = x_train.reshape(train_size, 28,28,1)
	x_test = x_test.reshape(x_test.shape[0], 28,28,1)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	x_train = x_train / 255
	x_test = x_test / 255

	return (x_train, y_train) , (x_test, y_test)

# args
train_size = 10000
batch_size = 200
epochs = 500

(x_train, y_train) , (x_test, y_test) = load_data(train_size)
model = [ Sequential(), Sequential(), Sequential(), Sequential() ]
hist = []

# model_1
model[0].add(Conv2D(9, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), use_bias=False))
model[0].add(MaxPooling2D(2,2))
model[0].add(Conv2D(6,(4,4), use_bias=False))
model[0].add(Conv2D(5,(3,3), use_bias=False))
model[0].add(Conv2D(28,(3,3), use_bias=False))
model[0].add(MaxPooling2D(2,2))

model[0].add(Flatten())
model[0].add(Dense(units=10, activation='relu'))
model[0].add(Dense(units=15, activation='relu'))
model[0].add(Dense(units=10, activation='softmax'))

# model_2
model[1].add(Conv2D(25, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), use_bias=False))
model[1].add(MaxPooling2D(2,2))
model[1].add(Conv2D(10,(3,3), use_bias=False))
model[1].add(MaxPooling2D(2,2))

model[1].add(Flatten())
model[1].add(Dense(units=10, activation='relu'))
model[1].add(Dense(units=16, activation='relu'))
model[1].add(Dense(units=10, activation='softmax'))

# model_3
model[2].add(Conv2D(25, kernel_size=(3,3), strides=(1,1), input_shape=(28,28,1), use_bias=False))
model[2].add(MaxPooling2D(2,2))
model[2].add(Conv2D(10,(3,3), use_bias=False))
model[2].add(MaxPooling2D(2,2))

model[2].add(Flatten())
model[2].add(Dense(units=10, activation='relu'))
model[2].add(Dense(units=7, activation='relu'))
model[2].add(Dense(units=12, activation='relu'))
model[2].add(Dense(units=7, activation='relu'))
model[2].add(Dense(units=10, activation='softmax'))

# model_4
model[3].add(Conv2D(17, kernel_size=(12,12), strides=(1,1), input_shape=(28,28,1), use_bias=False))
model[3].add(MaxPooling2D(4,4))

model[3].add(Flatten())
model[3].add(Dense(units=10, activation='relu'))
model[3].add(Dense(units=10, activation='softmax'))

for i in range(len(model)):
	model[i].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	hist.append( model[i].fit(x_train, y_train, batch_size=batch_size, epochs=epochs) )

for i in range(len(model)):
	print('\nModel_%i' %(i+1))
	print(model[i].summary())
	result = model[i].evaluate(x_train, y_train, batch_size=train_size)
	print('\nTrain acc: ', result[1])
	result = model[i].evaluate(x_test, y_test, batch_size=train_size)
	print('Test acc: ', result[1])

for i in range(len(model)):
	plt.plot(range(epochs), hist[i].history['acc'], label='model_%i' %(i+1))
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.yscale('log')
plt.legend()
plt.show()

for i in range(len(model)):
	plt.plot(range(epochs), hist[i].history['loss'], label='model_%i' %(i+1))
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.show()
