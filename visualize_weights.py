"""Visualize the weights."""
import os

import numpy as np

import matplotlib.pyplot as plt

import caffe

from utils import miniplaces_net, get_split


def vis_square(data):
    """
    Visualize data.

    Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size
       approx. sqrt(n) by sqrt(n)
    """
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1)) +     # add some space between filters
               ((0, 0),) * (data.ndim - 3))
    # don't pad the last dimension (if there is one)
    # pad with ones (white)
    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose(
            (0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')

fn = miniplaces_net(get_split('train'), train=True).name

log_dir = None

proto_fn = fn
param_fn = os.path.join(log_dir, 'place_net_iter_50000.caffemodel')

net = caffe.Net(proto_fn, param_fn, caffe.TEST)

filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
