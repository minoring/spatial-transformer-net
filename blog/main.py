import os 

import numpy as np 

from utils import *


def main():
  DIMS = (400, 400)

  # Load two cat images
  CAT1 = 'data/cat1.jpg'
  CAT2 = 'data/cat2.jpeg'

  img1 = img_to_array(CAT1, DIMS)
  img2 = img_to_array(CAT2, DIMS, view=True)

  # Concat into tensor of shape (2, 400, 400, 3)
  input_img = np.concatenate([img1, img2], axis=0)

  # Dimension sanity check
  print('INput img shape: {}'.format(input_img.shape))

  # Grab shape
  num_batch, H, W, C = input_img.shape

  # Initialize M to identity transform
  # M = np.array([[1., 0., 0.], [0., 1., 0.]])

  # Translate by 0.5 only in x direction
  M = np.array([[1., 0., 0.5], [0., 1., 0.]])

  # Rotate 45 degrees. Cos(45) = Sin(45) = 0.707
  # M = np.array([[0.707, -0.707, 0.], [0.707, 0.707, 0.]])

  # Repeat number_batch times
  M = np.resize(M, (num_batch, 2, 3))

  # Get grids
  batch_grids = affine_grid_generator(H, W, M)
  
  # Given x and y in the sampling grid, we want interpolate
  # the pixel value in the original image.

  # Seperate the x and y dimensions and rescaling them to belong
  # in the height/width interval
  x_s = batch_grids[:, :, :, 0:1].squeeze() # Remove single-dimensional entries.
  y_s = batch_grids[:, :, :, 1:2].squeeze()


  out = bilinear_sampler(input_img, x_s, y_s)
  print('Out Img shape: {}'.format(out.shape))

  # view the 2nd image
  array_to_img(out[-1]).show()
  array_to_img(out[0]).show()


  
if __name__ == '__main__':
  main()
