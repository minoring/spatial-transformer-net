# import tensorflow.compat.v1 as tf
import tensorflow as tf


def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
  """Spatial Transformer Layer

  Implementation a spatial transformer layer as described in [1]_.

  Args:
    U: float. The output of a convolutional net should have. the shape of
       (num_batch, height, width, num_channels)
    theta: float. The output of the localisation network should be (num_batch, 6).
    out_size: tuple of two ints. The size of the output of the network (height, width)
  
  References:
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
  Notes:
    To initialize the network to the identity transform init
    `theta` to: identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    identity = identity.flatten()
    theta = tf.Variable(initial_value=identity)
  """

  def _meshgrid(height, width):
    with tf.compat.v1.variable_scope('_meshgrid'):
      # This should be equivalent to:
      # x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
      #                        np.linspace(-1, 1, height))
      # ones = np.ones(np.prod(x_t.shape))
      # grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
      x_t = tf.matmul(
          tf.ones(shape=tf.stack([height, 1])),
          tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1),
                       [1, 0]))
      print(x_t)

      y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                tf.ones(shape=tf.stack([1, width])))
      print(y_t)
      # print(tf.ones(shape=height))
      # print(tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), axis=1), [1, 0]))


  def _transform(theta, input_dim, out_size):
    with tf.compat.v1.variable_scope('_transform'):
      num_batch = tf.shape(input_dim)[0]
      height = tf.shape(input_dim)[1]
      width = tf.shape(input_dim)[2]
      num_channels = tf.shape(input_dim)[3]

      theta = tf.reshape(theta, (-1, 2, 3))  # shape of (batch, 2, 3)
      theta = tf.cast(theta, 'float32')

      # Grid of (x_t, y_t, 1), eq (1) in ref [1]
      height_f = tf.cast(height, 'float32')
      width_f = tf.cast(width, 'float32')
      out_height = out_size[0]
      out_width = out_size[1]
      _meshgrid(out_height, out_width)
      # grid = _meshgrid(out_height, out_width)


  with tf.compat.v1.variable_scope(name):
    output = _transform(theta, U, out_size)


if __name__ == '__main__':
  transformer(tf.ones((1, 40, 40, 1)), tf.ones((1, 6)), (40, 40))