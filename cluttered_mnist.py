"""Implement STN in TF

Reference:
  https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/cluttered_mnist.py
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot

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
