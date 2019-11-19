# import tensorflow.compat.v1 as tf
import tensorflow as tf


def transformer(input_dim, theta, out_size, name='SpatialTransformer',
                **kwargs):
  """Spatial Transformer Layer

  Implementation a spatial transformer layer as described in [1]_.

  Args:
    input_dim:
      Input dimensions of CNN should have.
      The output of the layer preceding the localization network. 
      If the STN layer is the first layer of the network, 
      then this corresponds to the input images. Shape should be (B, H, W, C).
    theta: 
      The output of the localisation network should be (num_batch, 6).
    out_size:
      Desired (H, W) of the output of transformer.
      Useful for upsampling or downsampling.
      If not specified, then output dimensions will be equal to input_dim dimensions.
  
  References:
    .. [1] Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
  Notes:
    To initialize the network to the identity transform init
    `theta` to: identity = np.array([[1., 0., 0.], [0., 1., 0.]])
    identity = identity.flatten()
    theta = tf.Variable(initial_value=identity)
  """
  def _repeat(x, n_repeats):
      with tf.compat.v1.variable_scope('_repeat'):
          rep = tf.transpose(
              tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
          rep = tf.cast(rep, 'int32')
          x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
          return tf.reshape(x, [-1])

  def _interpolate(input_dim, x, y, out_size):
    with tf.compat.v1.variable_scope('_interpolate'):
      # Constants
      num_batch = tf.shape(input_dim)[0]
      height = tf.shape(input_dim)[1]
      width = tf.shape(input_dim)[2]
      channels = tf.shape(input_dim)[3]

      x = tf.cast(x, tf.float32)
      y = tf.cast(y, tf.float32)
      height_f = tf.cast(height, tf.float32)
      width_f = tf.cast(width, tf.float32)
      out_height = out_size[0]
      out_width = out_size[1]
      # Tensor value of 0. Same as tf.constant(0)
      zero = tf.zeros([], dtype=tf.int32)

      # Scale indices from [-1, 1] to [0, width of height]
      x = (x + 1.0) * width_f / 2.0
      y = (y + 1.0) * height_f / 2.0

      # Do sampling.
      x0 = tf.cast(tf.floor(x), tf.int32)
      x1 = x0 + 1
      y0 = tf.cast(tf.floor(y), tf.int32)
      y1 = y0 + 1

      # Clip out of range values of an image.
      max_y = tf.cast(tf.shape(input_dim)[1] - 1, tf.int32)
      max_x = tf.cast(tf.shape(input_dim)[2] - 1, tf.int32)
      x0 = tf.clip_by_value(x0, zero, max_x)
      x1 = tf.clip_by_value(x1, zero, max_x)
      y0 = tf.clip_by_value(y0, zero, max_y)
      y1 = tf.clip_by_value(y1, zero, max_y)

      dim2 = width
      dim1 = width * height
      base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
      base_y0 = base + y0 * dim2
      base_y1 = base + y1 * dim2
      print(base_y1)


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
      y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                      tf.ones(shape=tf.stack([1, width])))

      x_t_flat = tf.reshape(x_t, (1, -1))
      y_t_flat = tf.reshape(y_t, (1, -1))

      ones = tf.ones_like(x_t_flat)
      grid = tf.concat([x_t_flat, y_t_flat, ones], 0)

      return grid

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
      grid = _meshgrid(out_height, out_width)
      grid = tf.expand_dims(grid, 0)
      grid = tf.reshape(grid, [-1])  # Shape of (height * width * 3, ) = (4800,)
      # tf stack: convert rank 0 -> 1
      # tf.tile: Repeat batch of times.
      grid = tf.tile(grid, tf.stack([num_batch]))
      # tf.stack create tensor of (num_batch, 3, -1) values.
      # Same as grid = tf.reshape(grid, (num_batch, 3, -1))
      # (batch, 3, height * width) shape.
      grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

      # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
      T_g = tf.matmul(theta, grid)  # Shape of (batch, 2, height * width).
      x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])  # Shape (batch, 1, 1600)
      y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
      x_s_flat = tf.reshape(x_s, [-1])  # Shape of [-1] flattens into 1-D
      y_s_flat = tf.reshape(y_s, [-1])

      # input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
      _interpolate(input_dim, x_s_flat, y_s_flat, out_size)

      # output = tf.reshape(
      #     input_transformed,
      #     tf.stack([num_batch, out_height, out_width, num_channels]))

      # return output

  with tf.compat.v1.variable_scope(name):
    output = _transform(theta, input_dim, out_size)
    return output


if __name__ == '__main__':
  transformer(tf.ones((2, 40, 40, 1)), tf.ones((1, 6)), (40, 40))