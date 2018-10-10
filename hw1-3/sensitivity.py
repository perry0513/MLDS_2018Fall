import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.utils as np_utils
import tensorflow.keras.datasets.mnist as mnist
from matplotlib import pyplot as plt

def load_data(train_size, batch_size, batched):
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

	if batched == False:
		return (x_train, y_train), (x_test, y_test)

	# batched (train_size % batch_size == 0)
	train_data = list(zip( np.split(x_train, train_size/batch_size), np.split(y_train, train_size/batch_size) ))
	test_data  = list(zip( np.split(x_test , train_size/batch_size), np.split(y_test , train_size/batch_size) ))

	return train_data, test_data

# Parameters
train_size = 10000
epochs = 200
# train_size % batch_size == 0
batch_size = [ 50, 100, 125, 200, 250, 500, 
			   1000, 1250, 2000, 2500, 5000 ]	
display_step = 10

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
weights = [
	tf.Variable(tf.random_normal([num_input , n_hidden_1])),
	tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
	tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	tf.Variable(tf.random_normal([n_hidden_3, num_class ]))
]
biases = [
	tf.Variable(tf.random_normal([n_hidden_1])),
	tf.Variable(tf.random_normal([n_hidden_2])),
	tf.Variable(tf.random_normal([n_hidden_3])),
	tf.Variable(tf.random_normal([num_class ]))
]

# Create model
def neural_net(x):
	# Hidden fully connected layer with 256 neurons
	layer_1 = tf.nn.leaky_relu( tf.add(tf.matmul(x      , weights[0]), biases[0]) )
	# Hidden fully connected layer with 256 neurons
	layer_2 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_1, weights[1]), biases[1]) )
	# Hidden fully connected layer with 256 neurons
	layer_3 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_2, weights[2]), biases[2]) )
	# Output fully connected layer with a neuron for each class
	out_layer = tf.add( tf.matmul(layer_2, weights[3]), biases[3] )
	return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Calculate gradient of loss w.r.t. input
grad = tf.gradients(loss_op, X)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

train_losses, test_losses = [],[]
train_accs, test_accs = [],[]
sensitivity = []


for bs in batch_size:

	# Load training data
	train_data, test_data = load_data(train_size, bs, True)

	with tf.Session() as sess:
		sess.run(init)

		# Start training
		for epoch in range(epochs):
			# Shuffle batch
			np.random.shuffle(train_data)
			for step, (batch_x_train, batch_y_train) in enumerate(train_data):
				# Run optimization op (backprop)
				_, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={ X: batch_x_train, Y: batch_y_train })

				# Print progress
				if (step+1) % display_step == 0:
					print("Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | Train Loss: {:.6f} \t| Acc: {:.6f}"
						  .format(epoch+1, epochs, step+1, train_size/bs, loss/bs, acc) )


		# Calculate training/testing loss/acc
		(x_train, y_train), (x_test, y_test) = load_data(train_size, bs, False)
		train_loss, train_acc = sess.run([loss_op, accuracy], feed_dict={ X: x_train, Y: y_train })
		test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={ X: x_test, Y: y_test })

		train_losses.append(train_loss/train_size)
		train_accs.append(train_acc)
		test_losses.append(test_loss/train_size)
		test_accs.append(test_acc)


		# Get gradient of loss w.r.t. input
		grad_list = sess.run(grad, feed_dict={ X: x_train, Y: y_train })
		# Calculate square of gradient norm
		grad_norm = sum([ sum([ elem**2 for elem in g ]) for g in grad_list ][0])
		sensitivity.append(grad_norm ** 0.5)

# print(train_losses[0], train_accs[0], test_losses[0], test_accs[0], sensitivity[0])

# plot
fig1, ax1 = plt.subplots()
ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax1.set_xlabel('batch_size (log scale)')
ax1.set_xscale('log')
ax1.set_ylabel('cross_entropy', color=color)
ax1.set_yscale('log')
ax1.plot(batch_size, train_losses, color=color, label='train')
ax1.plot(batch_size, test_losses, color=color, label='test', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax2.set_ylabel('sensitivity', color=color)
ax2.plot(batch_size, sensitivity, color=color, label='sensitivity')
ax2.tick_params(axis='y', labelcolor=color)

fig1.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()



fig2, ax1 = plt.subplots()
ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax1.set_xlabel('batch_size (log scale)')
ax1.set_xscale('log')
ax1.set_ylabel('accuracy', color=color)
ax1.plot(batch_size, train_accs, color=color, label='train')
ax1.plot(batch_size, test_accs, color=color, label='test', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:red'
ax2.set_ylabel('sensitivity', color=color)
ax2.plot(batch_size, sensitivity, color=color, label='sensitivity')
ax2.tick_params(axis='y', labelcolor=color)

fig2.tight_layout()  # otherwise the right y-label is slightly clipped
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()
