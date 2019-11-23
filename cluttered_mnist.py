"""Implement STN in TF 2.0

Reference:
  https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/cluttered_mnist.py
"""
import tensorflow as tf
import numpy as np

from absl import app
from absl import flags

from spatial_transformer import transformer
from transformer import Transformer
from utils import create_gif
from utils import ImageCallback
from flags import define_flags


def run(flags_obj):
  # Load data
  mnist_cluttered = np.load('./data/mnist_sequence1_sample_5distortions5x5.npz')

  X_train = mnist_cluttered['X_train']  # (10000, 1600) shape.
  y_train = mnist_cluttered['y_train']
  X_valid = mnist_cluttered['X_valid']
  y_valid = mnist_cluttered['y_valid']
  X_test = mnist_cluttered['X_test']
  y_test = mnist_cluttered['y_test']

  X_train = np.reshape(X_train, (-1, 40, 40, 1))
  X_valid = np.reshape(X_valid, (-1, 40, 40, 1))

  BUFFER_SIZE = 10000
  BATCH_SIZE = 256
  train_ds = tf.data.Dataset.from_tensor_slices(
      (X_train, y_train)).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

  inputs = tf.keras.Input(shape=(
      40,
      40,
      1,
  ))
  locnet = tf.keras.layers.Flatten()(inputs)
  locnet = tf.keras.layers.Dense(20, input_shape=(40, 40, 1),
                                 activation='tanh')(locnet)
  locnet = tf.keras.layers.Dropout(rate=0.2)(locnet)
  locnet = tf.keras.layers.Dense(6, activation='tanh')(locnet)

  h_transformed = Transformer((40, 40))([inputs, locnet])

  h_transformed = tf.reshape(h_transformed, (-1, 40, 40, 1))
  conv_net = tf.keras.layers.Conv2D(16,
                                    3,
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu')(h_transformed)
  conv_net = tf.keras.layers.Conv2D(16,
                                    3,
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu')(conv_net)
  conv_net = tf.keras.layers.Flatten()(conv_net)
  conv_net = tf.keras.layers.Dense(1024, activation='relu')(conv_net)
  outs = tf.keras.layers.Dense(10)(conv_net)

  spatial_transformer_nets = tf.keras.Model(inputs=inputs, outputs=outs)

  # tf.keras.utils.plot_model(spatial_transformer_nets)

  # # Train in minibatches and report accuracy, loss
  EPOCHS = 300
  STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE

  spatial_transformer_nets.compile(
      optimizer='adam',
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=['accuracy'])

  spatial_transformer_nets.fit(
      train_ds,
      epochs=EPOCHS,
      steps_per_epoch=STEPS_PER_EPOCH,
      callbacks=[ImageCallback(X_train[:16])])
  
  create_gif()


def main(_):
  run(flags.Flag)


if __name__ == '__main__':
  define_flags()
  app.run(main)
