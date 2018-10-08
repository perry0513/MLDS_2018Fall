import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.utils as np_utils
import tensorflow.keras.datasets.mnist as mnist
from matplotlib import pyplot as plt

def load_data(train_size, batch_size):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[0:train_size]
	y_train = y_train[0:train_size]
	x_train = x_train.reshape(train_size, 28*28)
	x_test = x_test.reshape(x_test.shape[0], 28*28)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255
	x_test = x_test / 255
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	# batched (train_size % batch_size == 0)
	train_data = list(zip( np.split(x_train, train_size/batch_size), np.split(y_train, train_size/batch_size) ))
	test_data  = list(zip( np.split(x_test , train_size/batch_size), np.split(y_test , train_size/batch_size) ))

	return train_data, test_data

# Parameters
train_size = 10000
epochs = 100
batch_size = [ 100, 1000 ]	# train_size % batch_size == 0
display_step = 10
alpha_list = np.arange(-1,2,0.1)

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3nd layer number of neurons
num_input  = 784 # MNIST data input (img shape: 28*28)
num_class  = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_class])

# Store layers weight & bias
weights = [ [
    tf.Variable(tf.random_normal([num_input , n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    tf.Variable(tf.random_normal([n_hidden_3, num_class ]))
	], [
	tf.Variable(tf.random_normal([num_input , n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    tf.Variable(tf.random_normal([n_hidden_3, num_class ]))
	] ]
biases = [ [
    tf.Variable(tf.random_normal([n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_2])),
    tf.Variable(tf.random_normal([n_hidden_3])),
    tf.Variable(tf.random_normal([num_class ]))
	], [
    tf.Variable(tf.random_normal([n_hidden_1])),
    tf.Variable(tf.random_normal([n_hidden_2])),
    tf.Variable(tf.random_normal([n_hidden_3])),
    tf.Variable(tf.random_normal([num_class ]))
	] ]


# Create model
def neural_net(x, model_num):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu( tf.add(tf.matmul(x      , weights[model_num][0]), biases[model_num][0]) )
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_1, weights[model_num][1]), biases[model_num][1]) )
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_2, weights[model_num][2]), biases[model_num][2]) )
	# Output fully connected layer with a neuron for each class
    out_layer = tf.add( tf.matmul(layer_2, weights[model_num][3]), biases[model_num][3] )
    return out_layer

inter_train_loss, inter_train_acc = [],[]
inter_test_loss, inter_test_acc = [],[]
trained_model = []

for i in range(2):
	# Construct model
	logits = neural_net(X, i)
	prediction = tf.nn.softmax(logits)

	# Define loss and optimizer
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss_op)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# load training data
	train_data, test_data = load_data(train_size, batch_size[i])

	# Start training
	with tf.Session() as sess:
		sess.run(init)

		for epoch in range(epochs):
			# shuffle batch
			np.random.shuffle(train_data)
			for j, (batch_x_train, batch_y_train) in enumerate(train_data):
				# Run optimization op (backprop)
				_, train_loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={ X: batch_x_train, Y: batch_y_train })

				# print progress
				if (j+1) % display_step == 0:
					print("Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | Train Loss: {:.6f} \t| Acc: {:.6f}"
						  .format(epoch+1, epochs, j+1, train_size/batch_size[i], train_loss/batch_size[i], acc) )

		# record parameters, used to initialize inter_weights & inter_biases
		trained_model.append([ sess.run(weights[i]), sess.run(biases[i]) ])



for alpha in alpha_list:

	inter_weights = {
		'h1': tf.Variable(np.array(trained_model[0][0][0]) * alpha + np.array(trained_model[1][0][0]) * (1-alpha)),
		'h2': tf.Variable(np.array(trained_model[0][0][1]) * alpha + np.array(trained_model[1][0][1]) * (1-alpha)),
		'h3': tf.Variable(np.array(trained_model[0][0][2]) * alpha + np.array(trained_model[1][0][2]) * (1-alpha)),
		'out':tf.Variable(np.array(trained_model[0][0][3]) * alpha + np.array(trained_model[1][0][3]) * (1-alpha))
	}
	inter_biases = {
		'h1': tf.Variable(np.array(trained_model[0][1][0]) * alpha + np.array(trained_model[1][1][0]) * (1-alpha)),
		'h2': tf.Variable(np.array(trained_model[0][1][1]) * alpha + np.array(trained_model[1][1][1]) * (1-alpha)),
		'h3': tf.Variable(np.array(trained_model[0][1][2]) * alpha + np.array(trained_model[1][1][2]) * (1-alpha)),
		'out':tf.Variable(np.array(trained_model[0][1][3]) * alpha + np.array(trained_model[1][1][3]) * (1-alpha))
	}


	def inter_model(x):
		# Hidden fully connected layer with 256 neurons
	    layer_1 = tf.nn.leaky_relu( tf.add(tf.matmul(x      , inter_weights['h1']), inter_biases['h1']) )
	    # Hidden fully connected layer with 256 neurons
	    layer_2 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_1, inter_weights['h2']), inter_biases['h2']) )
	    # Hidden fully connected layer with 256 neurons
	    layer_3 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_2, inter_weights['h3']), inter_biases['h3']) )
		# Output fully connected layer with a neuron for each class
	    out_layer = tf.add( tf.matmul(layer_2, inter_weights['out']), inter_biases['out'] )
	    return out_layer

	# Construct inter_model
	logits = inter_model(X)
	prediction = tf.nn.softmax(logits)
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

	# Evaluate inter_model
	correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()

	# load testing data
	train_data, test_data = load_data(train_size, batch_size[0]) # batch_size = 100
	dataset = list(zip(train_data, test_data))

	# Start testing
	with tf.Session() as sess:
		sess.run(init)

		# Test model
		avg_train_loss, avg_train_acc = 0, 0
		avg_test_loss, avg_test_acc = 0, 0
		for i, ((batch_x_train, batch_y_train), (batch_x_test, batch_y_test)) in enumerate(dataset):
			# Calculate loss & accuracy
			train_loss, train_acc = sess.run([loss_op, accuracy], feed_dict={ X: batch_x_train, Y: batch_y_train })
			test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={ X: batch_x_test, Y: batch_y_test })

			avg_train_loss += train_loss/train_size
			avg_train_acc += train_acc/(train_size//batch_size[0])
			avg_test_loss += test_loss/train_size
			avg_test_acc += test_acc/(train_size//batch_size[0])

		print("Alpha: {:>5} | Train Loss: {:.6f} \t| Train Acc: {:.6f} | Test Loss: {:.6f} | Test Acc: {:.6f}"
			  .format(alpha, avg_train_loss, avg_train_acc, avg_test_loss, avg_test_acc))
		inter_train_loss.append(avg_train_loss)
		inter_train_acc.append(avg_train_acc)
		inter_test_loss.append(avg_test_loss)
		inter_test_acc.append(avg_test_acc)


# plot
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('alpha')
ax1.set_ylabel('cross_entropy', color=color)
ax1.set_yscale('log')
ax1.plot(alpha_list, inter_train_loss, color=color)
ax1.plot(alpha_list, inter_test_loss, color=color, linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('accuracy', color=color)  # already handled the x-label with ax1
ax2.plot(alpha_list, inter_train_acc, color=color, label='train')
ax2.plot(alpha_list, inter_test_acc, color=color, linestyle='--', label='test')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()
