import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.utils as np_utils
import tensorflow.keras.datasets.mnist as mnist
from matplotlib import pyplot as plt
# import PyQt5

def load_data(train_size):
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

	return (x_train, y_train), (x_test, y_test)

def shuffle_and_zip(x, y, ts, bs):
	zipped = list(zip(x,y))
	np.random.shuffle(zipped)
	x, y = [ np.array(t) for t in zip(*zipped) ]
	mod_ts = ts//bs * bs
	return list(zip( np.split(x[:mod_ts], mod_ts//bs), np.split(y[:mod_ts], mod_ts//bs) ))

# Parameters
train_size = 10000
test_size  = 10000
epochs = 2000
batch_size_list = range(50,2001,50)
display_step = 50
keep_prob = 0.5
e = 1e-4

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Create model
def neural_net(x):
	x_image = tf.reshape(x, [-1, 28, 28, 1])
   ## conv1 layer ##
	W_conv1 = weight_variable([3,3, 1,22])
	b_conv1 = bias_variable([22])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool(h_conv1,2)

	## conv2 layer ##
	W_conv2 = weight_variable([2,2, 22, 22])
	b_conv2 = bias_variable([22])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool(h_conv2,3)
	h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob)

	## fc1 layer ##
	W_fc1 = weight_variable([4*4*22, 80])
	b_fc1 = bias_variable([80])
	h_pool2_flat = tf.reshape(h_pool2_drop, [-1, 4*4*22])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	## fc2 layer ##
	W_fc2 = weight_variable([80, 10])
	b_fc2 = bias_variable([10])
	out_layer = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	return out_layer

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool(x, s):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,s,s,1], strides=[1,s,s,1], padding='VALID')

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

# Calculate Hessian matrix and sharpness
hessian = tf.hessians(loss_op,tf.trainable_variables())
hessian_arr = []
for h in hessian:
	hessian_arr.append(tf.norm(h,2))
hessian_norm = tf.reduce_max(hessian_arr)
sharpness = 0.5*tf.norm(hessian_norm,2)*e**2 / (1+loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Load training data
(x_train, y_train), (x_test, y_test) = load_data(train_size)

# Start training
sharpness_hist = []
train_loss_hist = []
test_loss_hist = []
for batch_size in batch_size_list:
	train_data = shuffle_and_zip(x_train, y_train, train_size, batch_size)
	test_data = shuffle_and_zip(x_test, y_test, test_size, batch_size)
	dataset = list(zip(train_data, test_data))
	total_step = train_size//batch_size

	with tf.Session() as sess:
		# Run the initializer
		sess.run(init)

		for epoch in range(epochs):
			np.random.shuffle(train_data)
			avg_train_loss, avg_test_loss = 0, 0
			for i, ((batch_x_train, batch_y_train), (batch_x_test, batch_y_test)) in enumerate(dataset):
				# Run optimization op (backprop)
				_, train_loss = sess.run([train_op, loss_op], feed_dict={ X: batch_x_train, Y: batch_y_train })
				test_loss = sess.run(loss_op, feed_dict={ X: batch_x_test, Y: batch_y_test })
				
				avg_train_loss += train_loss/total_step
				avg_test_loss += test_loss/total_step

				if (i+1==total_step) and (epoch+1==epochs):
					avg_sharpness = sess.run(sharpness, feed_dict={ X: batch_x_train, Y: batch_y_train })
					print (avg_sharpness)
					sharpness_hist.append(avg_sharpness)
					train_loss_hist.append(avg_train_loss)
					test_loss_hist.append(avg_test_loss)

# plot
plt.switch_backend('agg')
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xscale('log')
ax1.set_xlabel('batch size (log scale)')
ax1.set_ylabel('loss', color=color)
ax1.plot(batch_size_list, train_loss_hist, color=color, label='train')
ax1.plot(batch_size_list, test_loss_hist, color=color, linestyle='--', label='test')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('sharpness', color=color)  # already handled the x-label with ax1
ax2.plot(batch_size_list, sharpness_hist, color=color, label='sharpness')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.savefig('./bonus.png')
plt.show()

