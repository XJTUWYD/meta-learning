"""
Script to train MetaNet on mini-ImageNet.
Last few layers of MetaNet are fast-parameterized and layer-augmented for this task.

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

from generators import MiniImageNetGenerator

from MetaNetPartial import MetaNetPartial
from MetaNetPartialU import MetaNetPartialU

input_path = '/path/to/mini-imagenet/'


gpu = 0
n_epoch   = 100000   # number of epochs
batch_size = 1  # minibatch size
n_outputs = 10
n_outputs_test = 5
n_eposide = 10
n_eposide_tr = 100
nb_samples_per_class = 1
nb_samples_per_class_test = 15

print "batch_size", batch_size
print "GPU", gpu
print "n_classes (train):", n_outputs
print "n_classes (test):", n_outputs_test
print "nb_samples_per_class:", nb_samples_per_class
print "nb_samples_per_class_test:", nb_samples_per_class_test

print "train n_eposide:", n_eposide_tr
print "test n_eposide:", n_eposide

if gpu >= 0:
	chainer.cuda.get_device(gpu).use()

mod = cupy if gpu >= 0 else np

print "Loading train..."
train_generator = MiniImageNetGenerator(data_file=input_path + 'train.npz', 
									augment = False, mean_norm = False,
									nb_classes=n_outputs, nb_samples_per_class=1, 
									batchsize=batch_size, max_iter=None, xp=mod, 
									nb_samples_per_class_test=nb_samples_per_class_test)
mean = train_generator.mean

test_generator = MiniImageNetGenerator(data_file=input_path + 'test.npz', 
									augment = False, mean = mean, mean_norm = False,
									nb_classes=n_outputs_test, nb_samples_per_class=3, 
									batchsize=batch_size, max_iter=None, xp=mod, 
									nb_samples_per_class_test=nb_samples_per_class_test)

dev_generator = MiniImageNetGenerator(data_file=input_path + 'dev.npz', 
									augment = False, mean = mean, mean_norm = False,
									nb_classes=n_outputs_test, nb_samples_per_class=3, 
									batchsize=batch_size, max_iter=None, xp=mod, 
									nb_samples_per_class_test=nb_samples_per_class_test)


print "# train classes:", len(train_generator.data.keys())
print "# dev classes:", len(dev_generator.data.keys())
print "# test classes:", len(test_generator.data.keys())

max_dev = 0
max_test = 0
max_epoch = 0
print "Building model..."
model = MetaNetPartial(n_outputs, n_outputs_test, gpu)
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
		it, data = dev_generator.next()
		support_set, support_lbl, x_set, x_lbl = data
		preds_true.extend(x_lbl)
		preds_x, x_loss = model.predict(support_set, support_lbl, x_set, x_lbl)
		preds.extend(preds_x)
		loss_a += x_loss.data
	print "dev task loss:", loss_a/n_eposide
	acc = accuracy_score(preds_true, preds)
	print 'dev task accuracy_score={}'.format(acc)

	if max_dev <= acc:
		max_dev = acc
		max_epoch = i

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
		max_test = acc
	print "best epoch:", max_epoch
	print "best dev acc:", max_dev
	print "best test acc:", max_test