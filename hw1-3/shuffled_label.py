import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.utils as np_utils
import tensorflow.keras.datasets.mnist as mnist

def load_data(train_size, batch_size):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[0:train_size]
	y_train = y_train[0:train_size]
	x_train = x_train.reshape(train_size, 28*28)
	x_test = x_test.reshape(x_test.shape[0], 28*28)

	# shuffle label
	np.random.shuffle(y_train)
	np.random.shuffle(y_test)
	
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255
	x_test = x_test / 255
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	# batched
	train_data = list(zip( np.split(x_train, train_size/batch_size), np.split(y_train, train_size/batch_size) ))
	test_data = list(zip( np.split(x_test, train_size/batch_size), np.split(y_test, train_size/batch_size) ))

	return train_data, test_data

# Parameters
train_size = 10000
epochs = 200
batch_size = 100
display_step = 50

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1' : tf.Variable(tf.random_normal([num_input , n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_classes]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


train_data, test_data = load_data(train_size, batch_size)

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu( tf.add(tf.matmul(x, weights['h1']), biases['b1']) )
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) )
    # Output fully connected layer with a neuron for each class
    layer_3 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']) )

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
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

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
	sess.run(init)

	for epoch in range(epochs):
		for i, (batch_x, batch_y) in enumerate(train_data):
			# Run optimization op (backprop)
			sess.run(train_op, feed_dict={ X: batch_x, Y: batch_y })
			if (i+1) % display_step == 0:
				# Calculate batch loss and accuracy
				loss, acc = sess.run([loss_op, accuracy], feed_dict={ X: batch_x, Y: batch_y })
				print("Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | Loss: {:.6f} \t| Acc: {:.6f}"
					  .format(epoch+1, epochs, i+1, train_size//batch_size, loss, acc) )

	print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={X: mnist.test.images,
    #                                   Y: mnist.test.labels}))


