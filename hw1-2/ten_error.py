import time
from math import floor
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

# Load data
data = tf.keras.datasets.mnist.load_data(".mnist.npz")

(feature_train, label_train), (feature_test, label_test) = data

label_enc = LabelBinarizer()
y_train = label_enc.fit_transform(label_train).astype(np.float32)
x_train = feature_train.reshape((60000, 28 * 28)).astype(np.float32)
x_test = feature_test.reshape((-1, 28 * 28)).astype(np.float32)
y_test = label_test.astype(np.float32)

def function(arr):
	return 0.5*(np.sin(6*np.pi*(arr-0.5))+np.sin(8*np.pi*(arr-0.5)))
	# return np.exp(np.sin(40*arr))*np.log(arr+1)

# Parameters
data_size = 10000
learning_rate = 0.1
nb_epoch = 10000
batch_size = 100
show_epoch = 100

# np.random.seed(123)
# x_train = np.random.rand(data_size).reshape(data_size, 1)
# y_train = function(x_train).reshape(data_size, 1)
# x_test = np.random.rand(data_size).reshape(data_size, 1)
# y_test = function(x_test).reshape(data_size, 1)

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurons
num_classes = 10 # MNIST total classes (0-9 digits)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
y_true_test = tf.placeholder(tf.float32, [None])

# Create model
def neural_net(x):
    # Hidden fully connected layer with 10 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 10 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
train_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

correct_prediction = tf.equal(tf.cast(tf.argmax(logits, axis=1), tf.float32), y_true_test)
test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

n_batch = int(floor(x_train.shape[0] / batch_size))
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, nb_epoch+1):
        for b in tqdm(range(n_batch)):
            _, loss, train_acc = sess.run([train_op, loss_op, train_accuracy],
                                          feed_dict={X: x_train[b * batch_size: (b + 1) * batch_size],
                                                     Y: y_train[b * batch_size: (b + 1) * batch_size]})
        if i%show_epoch == 0:
            # print(y_train, y_test)
            print("Minibatch Loss= {:.4f}".format(loss) + ", Training Accuracy= {:.3f}".format(train_acc))
        

    test_acc = sess.run(test_accuracy, feed_dict={X: x_test, y_true_test: y_test})
    print(test_acc)

end = time.time()
print("Time cost: %f" % (end - start))
