import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.backend import get_session

import matplotlib.pyplot as plt


def load_data():
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		number = 10000
		x_train = x_train[0:number]
		y_train = y_train[0:number]
		x_train = x_train.reshape(number, 28*28)
		x_test = x_test.reshape(x_test.shape[0], 28*28)
		x_train = x_train.astype('float32')
		x_test = x_test.astype('float32')

		y_train = np_utils.to_categorical(y_train, 10)
		y_test = np_utils.to_categorical(y_test, 10)
		x_train = x_train
		x_test = x_test

		x_train = x_train / 255
		x_test = x_test / 255
		return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

#Add layers into the model
def add_params(model, layers):

	for x in range(2 * layers + 2):
		model.pop()

	model.set_weights(Wsave)
	

	for x in range(layers):
		model.add(Dense(units = units_per_layer))
		model.add(Activation('relu'))

	model.add(Dense(units = 10))
	model.add(Activation('softmax'))

	return model

'''
def reset_weights(model):
    session = get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
'''
def slightly_increase(x):
	x = x * 1.2
	return x

#Parameters
Epochs = 20
Batch_size = 100
units_per_layer = 1
units_per_layer_increase = 1
layers = 3
t = 100

param_hist = []
train_loss_hist, train_acc_hist = [],[]
test_loss_hist, test_acc_hist = [],[]
temp = []

#1_Define a set of function
model = Sequential()
model.add( Dense( input_dim = 28*28, units = 10))
model.add( Activation('relu'))
Wsave = model.get_weights()
for x in range(layers):
	model.add(Dense(units = units_per_layer))
	model.add(Activation('relu'))
model.add( Dense(units = 10))
model.add( Activation('softmax'))
print(model.summary())

#initial_weights = model.get_weights()
#l = np.log(3.1)

for i in range(t):
	print("\n\n=======================================================\nModel: ", i+1," / ",t,'\n')
	units_per_layer += units_per_layer_increase
#	print("\n!!!!!!!!", units_per_layer_increase)
	units_per_layer_increase =  units_per_layer_increase + int((i**1.15)/10 )
	model = add_params(model, layers)
	param_hist.append(model.count_params())

	model.compile(loss = 'categorical_crossentropy', optimizer = 'Adagrad', metrics = ['accuracy'])
	history = model.fit(x_train, y_train,validation_split=0.25, batch_size = Batch_size, epochs = Epochs)

	train_loss_hist.append(history.history['loss'][Epochs-1])
	train_acc_hist.append(history.history['acc'][Epochs-1])

	test_loss_hist.append(history.history['val_loss'][Epochs-1])
	test_acc_hist.append(history.history['val_acc'][Epochs-1])
	print(model.summary())


#	temp.append(history.history['loss'][0])


#	result = model.evaluate(x_test, y_test)
#	test_acc_hist.append(result[1])
	

'''
# Plot training & validation accuracy values
plt.plot(param_hist,temp)
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#plt.plot(param_hist,temp,'go')
'''
plt.plot(param_hist,'ro')
plt.show()



# Plot training & validation loss values
C1 = '#EE7700'
C2 = '#009FCC'
size = 5
plt.plot(param_hist, train_loss_hist, 'o', markersize = size, color = C1)
plt.plot(param_hist, test_loss_hist, 'o', markersize = size, color = C2)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Params')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

plt.plot(param_hist, train_acc_hist, 'o', markersize = size, color = C1)
plt.plot(param_hist, test_acc_hist, 'o', markersize = size, color = C2)
plt.title('Model Acc')
plt.ylabel('Acc')
plt.xlabel('Params')
plt.legend(['Train', 'Test'], loc='best')
plt.show()
