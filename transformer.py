import tensorflow as tf
from spatial_transformer import transformer


class Transformer(tf.keras.layers.Layer):

  def __init__(self, out_size):
    super(Transformer, self).__init__()

  def call(self, input_tensor, theta, out_size):
    output = transformer(input_tensor, theta, out_size)
    return output