import tensorflow as tf
from spatial_transformer import transformer


class Transformer(tf.keras.layers.Layer):

  def __init__(self, out_size):
    super(Transformer, self).__init__()
    self.out_size = out_size

  def call(self, tensors):
    input_img, theta = tensors
    output = transformer(input_img, theta, self.out_size)
    return output