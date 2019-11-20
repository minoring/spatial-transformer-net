"""Utility function to perform image"""
import matplotlib.pyplot as plt
import imageio
import glob
import os


def imshow(img):
  """Show Numpy array image ranged (-1, 1)"""
  out = img.copy()
  out = (out + 1.0) / 2.0
  plt.imshow(out[0, :, :, 0]) # Remove batch, color dimension
  plt.show()

def save_example_imgs(img, epoch):
  out = img.copy()
  # Normalize img
  out = (out + 1.0) / 2.0
  num_example = img.shape[0]

  num_row = 4
  num_col = num_example // num_row

  fig, axarr = plt.subplots(num_row, num_col)
  for i in range(num_row):
    for j in range(num_col):
      # axxarr[i, j].imshow()
      axarr[i, j].imshow(img[i * num_row + j, :, :, 0], cmap='binary')
      axarr[i, j].set_yticklabels([])
      axarr[i, j].set_xticklabels([])
      # plt.subplot(num_row, num_col, i * num_col + j + 1)
      # plt.imshow(img[i * num_row + j, :, :, 0], cmap='binary')
  
  if not os.path.isdir('samples'):
    os.mkdir('samples')
  plt.savefig('samples/epoch{}.jpg'.format(epoch))

  fig.clf()


def create_gif():
  """Create gif using saved images"""
  anim_file = 'cluttered.gif'

  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('samples/epoch*.jpg')
    filenames = sorted(filenames)
    last = -1
    for i, filename in enumerate(filenames):
      frame = 2 * (i**0.5)
      if round(frame) > round(last):
        last = frame
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)