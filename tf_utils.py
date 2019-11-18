import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def weight_variable(shape):
  """Helper function to create a weight variable initialized with zeros.

  Args:
    shape: list. Size of weight variable
  """
  initial = tf.zeros(shape)
  return tf.Variable(initial)


def bias_variable(shape):
  """Helper function to create a bias variable initialized with a normal distribution.
  
  Args:
    shape: list. Size of weight variable
  """
  initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
  return tf.Variable(initial)


def dense_to_one_hot(labels, n_classes=2):
  """Convert class labels from scalar to one-hot vectors."""
  labels = np.array(labels)
  n_labels = labels.shape[0]
  index_offset = np.arange(n_labels) * n_classes
  labels_one_hot = np.zeros((n_labels, n_classes), dtype=np.float32)
  labels_one_hot.flat[index_offset + labels.ravel()] = 1
  return labels_one_hot


if __name__ == '__main__':
  dense_to_one_hot([3, 4, 5, 1], n_classes=10)
