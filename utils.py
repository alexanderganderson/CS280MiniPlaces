"""Utilities for creating caffe networks."""

from __future__ import division

import os
import time
import tempfile
import argparse
import numpy as np
import sys

# tag = 'GLOG_minloglevel'
# if not os.environ.get(tag, ''):
os.environ['GLOG_minloglevel'] = '3'

import caffe
from caffe.proto import caffe_pb2
from caffe import layers
from caffe import params


weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

zero_filler = dict(type='constant', value=0)
msra_filler = dict(type='msra')
uniform_filler = dict(type='uniform', min=-0.1, max=0.1)
fc_filler = dict(type='gaussian', std=0.005)
conv_filler = dict(type='msra')

parser = argparse.ArgumentParser(
    description='Train and evaluate a net on the MIT mini-places dataset.')
parser.add_argument(
    '--image_root', default='./images/',
    help='Directory where images are stored')
parser.add_argument(
    '--crop', type=int, default=96,
    help=('The edge length of the random image crops'
          '(defaults to 96 for 96x96 crops)'))
parser.add_argument(
    '--disp', type=int, default=10,
    help='Print loss/accuracy every --disp training iterations')
parser.add_argument(
    '--snapshot_dir', default='./snapshot',
    help='Path to directory where snapshots are saved')
parser.add_argument(
    '--snapshot_prefix', default='place_net',
    help='Snapshot filename prefix')
parser.add_argument(
    '--iters', type=int, default=50 * 1000,
    help='Total number of iterations to train the network')
parser.add_argument(
    '--batch', type=int, default=256,
    help='The batch size to use for training')
parser.add_argument(
    '--iter_size', type=int, default=1,
    help=('The number of iterations (batches) over which to average the '
          'gradient computation. Effectively increases the batch size '
          '(--batch) by this factor, but without increasing memory use '))
parser.add_argument(
    '--lr', type=float, default=0.01,
    help='The initial learning rate')
parser.add_argument(
    '--gamma', type=float, default=0.1,
    help='Factor by which to drop the learning rate')
parser.add_argument(
    '--stepsize', type=int, default=10 * 1000,
    help='Drop the learning rate every N iters -- this specifies N')
parser.add_argument(
    '--momentum', type=float, default=0.9,
    help='The momentum hyperparameter to use for momentum SGD')
parser.add_argument(
    '--decay', type=float, default=5e-4,
    help='The L2 weight decay coefficient')
parser.add_argument(
    '--seed', type=int, default=1,
    help='Seed for the random number generator')
parser.add_argument(
    '--cudnn', action='store_true',
    help='Use CuDNN at training time -- usually faster, but non-deterministic')
parser.add_argument(
    '--gpu', type=int, default=0,
    help='GPU ID to use for training and inference (-1 for CPU)')


def get_split(split):
    """Get filename for split."""
    filename = './development_kit/data/%s.txt' % split
    if not os.path.exists(filename):
        raise IOError('Split data file not found: %s' % split)
    return filename


def to_tempfile(file_content):
    """Write protobuf to tempfile."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_content)
        return f.name


def snapshot_prefix(args):
    """Get prefix for snapshot directory."""
    return os.path.join(args.snapshot_dir, args.snapshot_prefix)


def snapshot_at_iteration(iteration, args):
    """Get name for snapshot file."""
    return '%s_iter_%d.caffemodel' % (snapshot_prefix(args), iteration)


def miniplaces_solver(train_net_path, args, test_net_path=None):
    """Get protobuf of miniplaces solver."""
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        # Test after every 1000 training iterations.
        s.test_interval = 1000
        # Set `test_iter` to test on 100 batches each time we test.
        # With test batch size 100, this covers the entire validation set of
        # 10K images (100 * 100 = 10K).
        s.test_iter.append(100)
    else:
        s.test_interval = args.iters + 1  # don't test during training

    # The number of batches over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = args.iter_size

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # The following settings (base_lr, lr_policy, gamma,
    # stepsize, and max_iter),
    # define the following learning rate schedule:
    #   Iterations [  0, 20K) -> learning rate 0.01   = base_lr
    #   Iterations [20K, 40K) -> learning rate 0.001  = base_lr * gamma
    #   Iterations [40K, 50K) -> learning rate 0.0001 = base_lr * gamma^2

    # Set the initial learning rate for SGD.
    s.base_lr = args.lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = args.gamma
    s.stepsize = args.stepsize

    # `max_iter` is the number of times to update the net
    # (training iterations).
    s.max_iter = args.iters

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help
    # prevent the model from overfitting.
    s.momentum = args.momentum
    s.weight_decay = args.decay

    # Display the current training loss and accuracy every
    # `display` iterations.
    # This doesn't have an effect for Python training here as logging is
    # disabled by this script (see the GLOG_minloglevel setting).
    s.display = args.disp

    # Number of training iterations over which to smooth the displayed loss.
    # The summed loss value (Iteration N, loss = X) will be averaged,
    # but individual loss values (Train net output #K: my_loss = X) won't be.
    s.average_loss = 10

    # Seed the RNG for deterministic results.
    # (May not be so deterministic if using CuDNN.)
    s.random_seed = args.seed

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot twice per learning rate step to the location specified by the
    # --snapshot_dir and --snapshot_prefix args.
    s.snapshot = args.stepsize // 2
    s.snapshot_prefix = snapshot_prefix(args)

    # Create snapshot dir if it doesn't already exist.
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    return to_tempfile(str(s))


def build_input(source, args, train):
    """Build input to network."""
    mean = [104, 117, 123]  # per-channel mean of the BGR image pixels
    transform_param = dict(mirror=train, crop_size=args.crop, mean_value=mean)
    batch_size = args.batch if train else 100
    places_data, places_labels = layers.ImageData(
        transform_param=transform_param,
        source=source, root_folder=args.image_root, shuffle=train,
        batch_size=batch_size, ntop=2)
    return places_data, places_labels


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=conv_filler, bias_filler=zero_filler,
              train=False, cudnn=True):
    """Get convolutional network."""
    # set CAFFE engine to avoid CuDNN convolution -- non-deterministic results
    engine = {}
    if train and not cudnn:
        engine.update(engine=params.Pooling.CAFFE)
    conv = layers.Convolution(
        bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, group=group, param=param,
        weight_filler=weight_filler, bias_filler=bias_filler,
        **engine)
    return conv, layers.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=fc_filler, bias_filler=zero_filler):
    """Add relu layer."""
    fc = layers.InnerProduct(
        bottom, num_output=nout, param=param,
        weight_filler=weight_filler, bias_filler=bias_filler)
    return fc, layers.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1, train=False, cudnn=True):
    """Add maxpooling layer."""
    # set CAFFE engine to avoid CuDNN pooling -- non-deterministic results
    engine = {}
    if train and not cudnn:
        engine.update(engine=params.Pooling.CAFFE)
    return layers.Pooling(
        bottom, pool=params.Pooling.MAX, kernel_size=ks,
        stride=stride, **engine)


def add_alexnet(n, top, train=False, param=learned_param,
                num_classes=100):
    """
    Return a protobuf text file specifying a variant of AlexNet.

     following the
    original specification (<caffe>/models/bvlc_alexnet/train_val.prototxt).
    The changes with respect to the original AlexNet are:
        - LRN (local response normalization) layers are not included
        - The Fully Connected (FC) layers (fc6 and fc7) have smaller dimensions
          due to the lower resolution of mini-places images (128x128) compared
          with ImageNet images (usually resized to 256x256)
    """
    # n = caffe.NetSpec()
    # n.data = data
    conv_kwargs = dict(param=param, train=train)

    dim = 96
    print 'Input dim is {}'.format(dim)

    fsize_ = [11, 5, 3, 3, 3]
    nout_ = [96, 256, 384, 384, 256]
    stride_ = [4, 1, 1, 1, 1]
    group_ = [1, 2, 1, 2, 2]
    pool_ = [True, True, False, False, True]

    for i, (fsize, nout, stride, pool, group) in enumerate(
            zip(fsize_, nout_, stride_, pool_, group_)):
        if i is 1:
            pad = 2
        else:
            pad = 1
        conv, relu = conv_relu(
            top, fsize, nout, stride=stride,
            pad=pad, group=group, **conv_kwargs)
        setattr(n, 'conv{}'.format(i), conv)
        setattr(n, 'relu{}'.format(i), relu)
        top = relu
        dim = int((dim - fsize + 1 + 2 * pad) / stride)
        print 'Dim after convolution {} = {}'.format(i, dim)
        if i is 0:
            nin = 3
        else:
            nin = nout_[i - 1]
        print 'Number of parameters is {:10}'.format(
            nout * nin * fsize ** 2 / group)

        if pool:
            pl = max_pool(top, 3, stride=2, train=train)
            setattr(n, 'pool{}'.format(i), pl)
            top = pl
            dim = int((dim - 3 + 1 + 2) / 2)
            print 'Dim after pooling {} is {}'.format(i, dim)
    sys.stdout.flush()

    nh_ = [1024]
    rlu_ = [True]
    drp_ = [True]
    for i, (nh, rlu, drp) in enumerate(nh_, rlu_, drp_):
        i += len(fsize_)
        top = fc = layers.InnerProduct(
            top, num_output=nh, param=param, weight_filler=fc_filler,
            bias_filler=zero_filler)
        setattr(n, 'fc{}'.format(i), fc)
        if rlu:
            top = relu = layers.ReLU(fc, in_place=True)
            setattr(n, 'relu{}'.format(i), relu)
        if drp:
            top = drop = layers.Dropout(top, in_place=True)
            setattr(n, 'drop{}'.format(i), drop)
        # n.fc6, n.relu6 = fc_relu(top, 1024, param=param)
        # n.drop6 = layers.Dropout(n.relu6, in_place=True)

    n.fc7, n.relu7 = fc_relu(top, 1024, param=param)
    n.drop7 = layers.Dropout(n.relu7, in_place=True)
    n.fc8 = layers.InnerProduct(
        n.drop7, num_output=num_classes, param=param)
    top = n.fc8
    return top


def build_test_train(n, top, train, with_labels, labels):
    """Take in current netspec and top, and adds final layers."""
    if train:
        preds = top
    else:
        # Compute the per-label probabilities at test/inference time.
        preds = n.probs = layers.Softmax(top)
    if with_labels:
        n.label = labels
        n.loss = layers.SoftmaxWithLoss(n.fc8, n.label)
        n.accuracy_at_1 = layers.Accuracy(preds, n.label)
        n.accuracy_at_5 = layers.Accuracy(
            preds, n.label, accuracy_param=dict(top_k=5))
    else:
        n.ignored_label = labels
        n.silence_label = layers.Silence(n.ignored_label, ntop=0)
    # return to_tempfile(str(n.to_proto()))


def miniplaces_net(source, args, train=False, with_labels=True):
    """Create a prototxt file for the network."""
    n = caffe.NetSpec()
    places_data, places_labels = build_input(source, args, train)
    top = n.data = places_data
    top = add_alexnet(n, top, train=train)
    build_test_train(n, top, train, with_labels, places_labels)
    return to_tempfile(str(n.to_proto()))


def train_net(args, with_val_net=False):
    """Train the network."""
    train_net_file = miniplaces_net(get_split('train'), args, train=True)
    # Set with_val_net=True to test during training.
    # Environment variable GLOG_minloglevel should be set to 0 to display
    # Caffe output in this case; otherwise, the test result will not be
    # displayed.
    if with_val_net:
        val_net_file = miniplaces_net(get_split('val'), args, train=False)
    else:
        val_net_file = None
    solver_file = miniplaces_solver(train_net_file, args, val_net_file)
    solver = caffe.get_solver(solver_file)
    outputs = sorted(solver.net.outputs)

    def str_output(output):
        value = solver.net.blobs[output].data
        if output.startswith('accuracy'):
            valstr = '%5.2f%%' % (100 * value, )
        else:
            valstr = '%6f' % value
        return '%s = %s' % (output, valstr)

    def disp_outputs(iteration, iter_pad_len=len(str(args.iters))):
        metrics = '; '.join(str_output(o) for o in outputs)
        return 'Iteration %*d: %s' % (iter_pad_len, iteration, metrics)
    # We could just call `solver.solve()` rather than `step()`ing in a loop.
    # (If we hadn't set GLOG_minloglevel = 3 at the top of this file, Caffe
    # would display loss/accuracy information during training.)
    previous_time = None
    for iteration in xrange(args.iters):
        solver.step(1)
        if (args.disp > 0) and (iteration % args.disp == 0):
            current_time = time.clock()
            if previous_time is None:
                benchmark = ''
            else:
                time_per_iter = (current_time - previous_time) / args.disp
                benchmark = ' (%5f s/it)' % time_per_iter
            previous_time = current_time
            print disp_outputs(iteration), benchmark
    # Print accuracy for last iteration.
    solver.net.forward()
    disp_outputs(args.iters)
    solver.net.save(snapshot_at_iteration(args.iters, args))


def eval_net(split, n_k=5):
    """Evaluate the network for a given split."""
    print 'Running evaluation for split:', split
    filenames = []
    labels = []
    split_file = get_split(split)
    with open(split_file, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            assert 1 <= len(parts) <= 2, 'malformed line'
            filenames.append(parts[0])
            if len(parts) > 1:
                labels.append(int(parts[1]))
    known_labels = (len(labels) > 0)
    if known_labels:
        assert len(labels) == len(filenames)
    else:
        # create file with 'dummy' labels (all 0s)
        split_file = to_tempfile(
            ''.join('%s 0\n' % name for name in filenames))
    test_net_file = miniplaces_net(split_file, args,
                                   train=False, with_labels=False)
    weights_file = snapshot_at_iteration(args.iters, args)
    net = caffe.Net(test_net_file, weights_file, caffe.TEST)
    top_k_predictions = np.zeros((len(filenames), n_k), dtype=np.int32)
    if known_labels:
        correct_label_probs = np.zeros(len(filenames))
    offset = 0
    while offset < len(filenames):
        probs = net.forward()['probs']
        for prob in probs:
            top_k_predictions[offset] = (-prob).argsort()[:n_k]
            if known_labels:
                correct_label_probs[offset] = prob[labels[offset]]
            offset += 1
            if offset >= len(filenames):
                break
    if known_labels:
        def accuracy_at_k(preds, labels, k):
            assert len(preds) == len(labels)
            num_correct = sum(l in p[:k] for p, l in zip(preds, labels))
            return num_correct / len(preds)
        for k in [1, n_k]:
            accuracy = 100 * accuracy_at_k(top_k_predictions, labels, k)
            print '\tAccuracy at %d = %4.2f%%' % (k, accuracy)
        cross_ent_error = -np.log(correct_label_probs).mean()
        print '\tSoftmax cross-entropy error = %.4f' % (cross_ent_error, )
    else:
        print 'Not computing accuracy; ground truth unknown for split:', split
    filename = 'top_%d_predictions.%s.csv' % (n_k, split)
    with open(filename, 'w') as f:
        f.write(','.join(['image'] +
                         ['label%d' % i for i in range(1, n_k + 1)]))
        f.write('\n')
        f.write(''.join('%s,%s\n' % (image, ','.join(str(p) for p in preds))
                        for image, preds in zip(filenames, top_k_predictions)))
    print 'Predictions for split %s dumped to: %s' % (split, filename)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gpu >= 0:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu)
    else:
        caffe.set_mode_cpu()

    train_net(args)
    print '\nTraining complete. Evaluating...\n'
    for split in ('train', 'val', 'test'):
        eval_net(split)
        print
    print 'Evaluation complete.'
