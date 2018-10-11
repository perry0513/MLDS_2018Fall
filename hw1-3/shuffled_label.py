import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as np_utils
import tensorflow.keras.datasets.mnist as mnist
from matplotlib import pyplot as plt

def load_data(train_size):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train[0:train_size]
	y_train = y_train[0:train_size]
	x_train = x_train.reshape(train_size, 28*28)
	x_test = x_test.reshape(x_test.shape[0], 28*28)

	# shuffle label of training data
	np.random.shuffle(y_train)
	# np.random.shuffle(y_test)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train = x_train / 255
	x_test = x_test / 255
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	return (x_train, y_train), (x_test, y_test)

def shuffle_and_zip(x, y, ts, bs):
	zipped = list(zip(x,y))
	np.random.shuffle(zipped)
	x, y = [ np.array(t) for t in zip(*zipped) ]
	mod_ts = ts//bs * bs
	return list(zip( np.split(x[:mod_ts], mod_ts//bs), np.split(y[:mod_ts], mod_ts//bs) ))


# Parameters
train_size = 10000
epochs = 200
batch_size = 100	# train_size % batch_size == 0
display_step = 50

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_hidden_3 = 256 # 3nd layer number of neurons
num_input  = 784 # MNIST data input (img shape: 28*28)
num_class  = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_class])

# Store layers weight & bias
weights = {
    'h1' : tf.Variable(tf.random_normal([num_input , n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, num_class ]))
}
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
    'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_class ]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu( tf.add(tf.matmul(x      , weights['h1']), biases['b1']) )
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']) )
    # Hidden fully connected layer with 256 neurons
    layer_3 = tf.nn.leaky_relu( tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']) )
    # Output fully connected layer with a neuron for each class
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

# load data, zip training data into "train_data"
(x_train, y_train), (x_test, y_test) = load_data(train_size)

train_loss_hist = []
test_loss_hist = []
# Start training
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(epochs):
		# shuffle batch
		train_data = shuffle_and_zip(x_train, y_train, train_size, batch_size)
		for batch_x_train, batch_y_train in train_data:
			# Run optimization op (backprop)
			 sess.run(train_op, feed_dict={ X: batch_x_train, Y: batch_y_train })
		
		# Calculate loss & acc
		train_loss, acc = sess.run([loss_op, accuracy], feed_dict={ X: x_train, Y: y_train })
		test_loss = sess.run(loss_op, feed_dict={ X: x_test, Y: y_test })

		print("Epoch [{:>3}/{:>3}] | Train Loss: {:.6f} | Acc: {:.6f} \t| Test Loss: {:.6f}"
			  .format(epoch+1, epochs, train_loss/train_size, acc, test_loss/train_size) )

		train_loss_hist.append(train_loss/train_size)
		test_loss_hist.append(test_loss/train_size)

# plot
plt.plot(train_loss_hist, label='train')
plt.plot(test_loss_hist, label='test')
plt.title('shuffled label')
plt.legend()
plt.show()

