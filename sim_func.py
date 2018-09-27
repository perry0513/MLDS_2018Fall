import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from keras.callbacks import Callback
from matplotlib import pyplot as plt

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# function to be simulated
def function(arr):
	# return 0.5*np.sin(10*np.pi*(arr-0.5))*np.abs(arr-0.5)
	# return 0.5*np.sin(10*np.pi*(arr-0.5))/(10*np.pi*(arr-0.5))
	return 0.5*(np.sin(6*np.pi*(arr-0.5))+np.sin(8*np.pi*(arr-0.5)))

# args
data_size = 10000
batch_size = 300
epochs = 500

# training and testing dataset
np.random.seed(123)
x_train = np.random.rand(data_size)
y_train = function(x_train)
x_test = np.random.rand(data_size)
y_test = function(x_test)

model = [Sequential(), Sequential(), Sequential(), Sequential()]
hist = []

# model_1 
model[0].add(Dense(input_dim=1, units=5))
model[0].add(LeakyReLU(alpha=0.1))
for i in range(11):
	model[0].add(Dense(units=10))
	model[0].add(LeakyReLU(alpha=0.1))
model[0].add(Dense(units=5))
model[0].add(LeakyReLU(alpha=0.1))
model[0].add(Dense(units=4))
model[0].add(LeakyReLU(alpha=0.1))
model[0].add(Dense(units=1))

# model_2 
model[1].add(Dense(input_dim=1, units=4))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=9))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=16))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=25))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=16))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=9))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=4))
model[1].add(LeakyReLU(alpha=0.1))
model[1].add(Dense(units=1))

# model_3
model[2].add(Dense(input_dim=1, units=18))
model[2].add(LeakyReLU(alpha=0.1))
model[2].add(Dense(units=32))
model[2].add(LeakyReLU(alpha=0.1))
model[2].add(Dense(units=18))
model[2].add(LeakyReLU(alpha=0.1))
model[2].add(Dense(units=1))


# model_4
model[3].add(Dense(input_dim=1, units=417))
model[3].add(LeakyReLU(alpha=0.1))
model[3].add(Dense(units=1))


for i in range(len(model)):
	model[i].compile(loss='mse', optimizer='adam')
	hist.append( model[i].fit(x_train, y_train, batch_size=batch_size, epochs=epochs) )

for i in range(len(model)):
	print('\nModel_%i' % (i+1) )
	print(model[i].summary())
	result = model[i].evaluate(x_train, y_train, batch_size=data_size)
	print('Train loss: %f' % result)
	result = model[i].evaluate(x_test, y_test, batch_size=data_size)
	print('Test loss: %f' % result)

# plot
with np.errstate(invalid='ignore', divide='ignore'):
	x_plot = np.arange(0.0, 1.0, 0.00001)
	y_plot = function(x_plot)
	for i in range(len(model)): 
		plt.plot(x_plot, model[i].predict(x_plot), label='model_'+str(i+1))
	plt.plot(x_plot, y_plot, label='function')
	plt.title('sim func')
	plt.xlabel('x')
	plt.ylabel('y')	
	plt.legend()
	plt.show()

	for i in range(len(model)):
		plt.plot(range(epochs), hist[i].history['loss'], label='model_'+str(i+1)+'_loss')
	plt.title('model loss')
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.yscale('log')
	plt.legend()
	plt.show()


	print(model[i].summary())
