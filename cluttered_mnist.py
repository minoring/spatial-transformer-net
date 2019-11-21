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
from tf_utils import dense_to_one_hot
from utils import imshow
from utils import save_example_imgs
from utils import create_gif
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

  train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
  test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

  BATCH_SIZE = 256
  train_ds.shuffle(10000).batch(batch_size=BATCH_SIZE)

  inputs = tf.keras.Input(shape=(40, 40, 1,))

  localisation_net = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(20, input_shape=(40, 40, 1), activation='tanh'),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(6, activation='tanh')
  ])

  h_transformed = transformer(inputs, localisation_net(inputs), (40, 40))

  conv_net = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(40, 40, 1)),
      tf.keras.layers.Conv2D(16,
                             3,
                             strides=(2, 2),
                             padding='same',
                             activation='relu'),
      tf.keras.layers.Conv2D(16,
                             3,
                             strides=(2, 2),
                             padding='same',
                             activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  out = conv_net(h_transformed)

  spatial_transformer_nets = tf.keras.Model(inputs=train_ds, outputs=out)

  spatial_transformer_nets(train_ds)

  # tf.keras.utils.plot_model(spatial_transformer_nets)
  
  # # Train in minibatches and report accuracy, loss
  # n_epochs = 300
  # steps_per_epoch = 10000 // BATCH_SIZE
  # spatial_transformer_nets.fit(train_ds,
  #                              steps_per_epoch=steps_per_epoch,
  #                              epochs=n_epochs)

  # save_example_imgs(example_img_transformed, epoch_i)
  # create_gif()


def main(_):
  run(flags.Flag)


if __name__ == '__main__':
  define_flags()
  app.run(main)
