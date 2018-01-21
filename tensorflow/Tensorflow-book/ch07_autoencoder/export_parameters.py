from autoencoder import Autoencoder
from scipy.misc import imread, imresize, imsave
import numpy as np
import h5py

def zero_pad(num, pad):
    return format(num, '0' + str(pad))

data_dir = '../vids/'
filename_prefix = 'raw_rgb_'

hidden_dim = 1000

filepath = data_dir + str(1) + '/' + filename_prefix + zero_pad(20, 5) + '.png'
img = imresize(imread(filepath, True), 1. / 8.)

img_data = img.flatten()

ae = Autoencoder([img_data], hidden_dim)

weights, biases = ae.get_params()

print(np.shape(weights))
print(np.shape([biases]))

h5f_W = h5py.File('encoder_W.h5', 'w')
h5f_W.create_dataset('dataset_1', data=weights)
h5f_W.close()

h5f_b = h5py.File('encoder_b.h5', 'w')
h5f_b.create_dataset('dataset_1', data=[biases])
h5f_b.close()
