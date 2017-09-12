"""
Script to train MetaNet on Omniglot.
The full MetaNet model is fast-parameterized and layer-augmented for this task.

"""
import argparse
import itertools
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import cupy

import numpy as np
from sklearn.metrics import accuracy_score

from generators import OmniglotGenerator

from MetaNetFull import MetaNetFull

input_path = '/path/to/omniglot/'

gpu = 0
n_epoch   = 100000   # number of epochs
batch_size = 1  # minibatch size
n_outputs_train = 5
n_outputs_test = 5
n_eposide = 10
n_eposide_tr = 100
nb_samples_per_class = 1
nb_samples_per_class_train = 10
nb_samples_per_class_test = 10

print "batch_size", batch_size
print "GPU", gpu
print "n_classes train:", n_outputs_train
print "n_classes test:", n_outputs_test
print "nb_samples_per_class:", nb_samples_per_class
print "nb_samples_per_class_train:", nb_samples_per_class_train
print "nb_samples_per_class_test:", nb_samples_per_class_test

print "train n_eposide:", n_eposide_tr
print "test n_eposide:", n_eposide

if gpu >= 0:
	chainer.cuda.get_device(gpu).use()

mod = cupy if gpu >= 0 else np

print "Loading train..."
train_generator = OmniglotGenerator(data_file=input_path + 'omniglot-data/train_rand.npz', 
									augment = True,
									nb_classes=n_outputs_train, nb_samples_per_class=nb_samples_per_class, 
									batchsize=batch_size, max_iter=None, xp=mod, 
									nb_samples_per_class_test=nb_samples_per_class_train)

test_generator = OmniglotGenerator(data_file=input_path + 'omniglot-data/test_rand.npz', 
									augment = False,
									nb_classes=n_outputs_test, nb_samples_per_class=nb_samples_per_class, 
									batchsize=batch_size, max_iter=None, xp=mod, 
									nb_samples_per_class_test=nb_samples_per_class_test)

print "# train classes:", len(train_generator.data.keys())
print "# test classes:", len(test_generator.data.keys())

print "Building model..."
model = MetaNetFull(n_outputs_train, gpu)
print "model:",model
model.init_optimizer()
print "Train looping..."
for i in xrange(0, n_epoch):
	print '======================='
	print "epoch={}".format(i)
	preds = []
	preds_true = []
	loss_a = 0
	begin_time = time.time()
	for j in xrange(n_eposide_tr):
		it, data = train_generator.next()
		support_set, support_lbl, x_set, x_lbl = data
		preds_true.extend(x_lbl)
		preds_x, x_loss = model.train(support_set, support_lbl, x_set, x_lbl)
		preds.extend(preds_x)
		loss_a += x_loss.data
	print 'secs per train epoch={}'.format(time.time() - begin_time)
	print "train task loss:", loss_a/n_eposide_tr
	acc = accuracy_score(preds_true, preds)
	print 'train task accuracy_score={}'.format(acc)
	preds = []
	preds_true = []
	loss_a = 0
	for j in xrange(n_eposide):
		it, data = test_generator.next()
		support_set, support_lbl, x_set, x_lbl = data
		preds_true.extend(x_lbl)
		preds_x, x_loss = model.predict(support_set, support_lbl, x_set, x_lbl)
		preds.extend(preds_x)
		loss_a += x_loss.data
	print "test task loss:", loss_a/n_eposide
	acc = accuracy_score(preds_true, preds)
	print 'test task accuracy_score={}'.format(acc)