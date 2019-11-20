"""Implement STN in TF 2.0

Reference:
  https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/cluttered_mnist.py
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from spatial_transformer import transformer
from tf_utils import weight_variable
from tf_utils import bias_variable
from tf_utils import dense_to_one_hot
from utils import imshow
from utils import save_example_imgs
from utils import create_gif 


# Load data
mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

X_train = mnist_cluttered['X_train']  # (10000, 1600) shape.
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# Turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

# Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])

# Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph. If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant. Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

# We will setup the two-layer localisation network to figure out the
# parameters for an affine transformation of the input
# Create variables for fully connected layer
W_fc_loc1 = weight_variable([1600, 20])  # TODO. keras
b_fc_loc1 = bias_variable([20])

W_fc_loc2 = weight_variable([20, 6])
# Use identity transformation as starting point
initial = np.array([[1., 0., 0.], [0., 1., 0.]]).astype('float32')
initial = initial.flatten()
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

# Define the two layer localisation network
h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
# We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
# Second layer
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

# We will create a spatial transformer module to identify discriminative
# patches
out_size = (40, 40)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)

# Weight matrix shape (filter_height, filter_width, input_channels, output_channels)
filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

# Bias is (output_channels)
b_conv1 = bias_variable([n_filters_1])

# Build a graph which does the first layer of convolution:
# We define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(
        input=h_trans, filter=W_conv1, strides=[1, 2, 2, 1], padding='SAME') +
    b_conv1)

n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(
        input=h_conv1, filters=W_conv2, strides=[1, 2, 2, 1], padding='SAME') +
    b_conv2)

# Reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])

# Create a fully-connected layer
n_fc = 1024
W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Finally, softmax layer
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))

opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
# Why localisation layer?
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

# Monitor accuracy
correct_prediction = tf.math.equal(tf.math.argmax(y_logits, axis=1),
                                   tf.math.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialization the variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train in minibatches and report accuracy, loss
iter_per_epoch = 100
n_epochs = 300
target_size = 10000

indices = np.linspace(0, 10000 - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
  for iter_i in range(iter_per_epoch - 1):
    batch_xs = X_train[indices[iter_i]:indices[iter_i + 1]]
    batch_ys = Y_train[indices[iter_i]:indices[iter_i + 1]]

    # if iter_i % 20 == 0:
    #   loss = sess.run(cross_entropy,
    #                   feed_dict={
    #                       x: batch_xs,
    #                       y: batch_ys,
    #                       keep_prob: 1.0
    #                   })
    #   print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch_i, iter_i, loss))

    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})

  print(
      'Accuracy epoch %d: ' % epoch_i +
      str(sess.run(accuracy, feed_dict={
          x: X_valid,
          y: Y_valid,
          keep_prob: 1.0
      })))

  if epoch_i % 10 == 0:
    example_img_transformed = sess.run(h_trans, feed_dict={
        x: X_train[0:16],
        y: Y_train[0:16],
        keep_prob: 1.0
    })
    save_example_imgs(example_img_transformed, epoch_i)

create_gif()