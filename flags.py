from absl import flags


def define_flags():
  flags.DEFINE_integer('batch_size', 256, 'The size of batch [256]')
  flags.DEFINE_integer('epoch', 150, 'Epoch to train [150]')
  flags.DEFINE_integer('locnet_height', 40,
                       'The height of output image of localisation networks')
  flags.DEFINE_integer('locnet_width', 40,
                       'The height of output image of localisation networks')
