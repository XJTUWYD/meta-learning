"""
Generators of data

This code comes from https://github.com/tristandeleu/ntm-one-shot
and was modified
"""
import numpy as np
import os
import random
import codecs

from images import rotate_right

import chainer

class OmniglotGenerator(object):
	"""OmniglotGenerator

	Args:
		data_file (str): 'data/omniglot/train.npz' or 'data/omniglot/test.npz' 
		nb_classes (int): number of classes in an episode
		nb_samples_per_class (int): nuber of samples per class in an episode
		batchsize (int): number of episodes in each mini batch
		max_iter (int): max number of episode generation
		xp: numpy or cupy
	"""
	def __init__(self, data_file, augment=True, nb_classes=5, nb_samples_per_class=10, 
				 batchsize=64, max_iter=None, xp=np, nb_samples_per_class_test=2):
		super(OmniglotGenerator, self).__init__()
		self.data_file = data_file
		self.nb_classes = nb_classes
		self.nb_samples_per_class = nb_samples_per_class
		self.nb_samples_per_class_test = nb_samples_per_class_test
		self.batchsize = batchsize
		self.max_iter = max_iter
		self.xp = xp
		self.num_iter = 0
		self.augment = augment
		self.data = self._load_data(self.data_file, self.xp)

	def _load_data(self, data_file, xp):
		data_dict = np.load(data_file)
		return {key: np.array(val) for (key, val) in data_dict.items()}

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def next(self):
		if (self.max_iter is None) or (self.num_iter < self.max_iter):
			labels_and_images_support = []
			labels_and_images_test = []

			self.num_iter += 1
			sampled_characters = random.sample(self.data.keys(), self.nb_classes) # list of keys

			for _ in xrange(self.batchsize):
				support_set = []

				for k,char in enumerate(sampled_characters):
					deg = random.sample(range(4), 1)[0]
					_imgs = self.data[char]
					_ind = random.sample(range(len(_imgs)), self.nb_samples_per_class+self.nb_samples_per_class_test)
					_ind_tr = _ind[:self.nb_samples_per_class]
					if self.augment:
						support_set.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_tr])
					else:
						support_set.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_tr])
					_ind_test = _ind[self.nb_samples_per_class:]
					if self.augment:
						labels_and_images_test.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_test])
					else:
						labels_and_images_test.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_test])

				random.shuffle(support_set)
				labels_tr, images_tr = zip(*support_set)

				labels_and_images_support.append((images_tr, labels_tr))

			_images, _labels = zip(*labels_and_images_support)
			images_tr = [
				self.xp.concatenate(map(lambda x: x.reshape((1,-1)), _img), axis=0)
				for _img in zip(*_images)]
			labels_tr = [_lbl for _lbl in zip(*_labels)]
			
			random.shuffle(labels_and_images_test)
			labels_test, images_test = zip(*labels_and_images_test)
			images_test = self.xp.concatenate([img.reshape((1, -1)) for img in images_test], axis=0)

			return (self.num_iter - 1), (images_tr, labels_tr, images_test, labels_test)
		else:
			raise StopIteration()

	def sample(self, nb_classes, nb_samples_per_class, nb_samples_per_class_test):
		sampled_characters = random.sample(self.data.keys(), nb_classes) # list of keys
		labels_and_images_support = []
		labels_and_images_test = []
		for (k, char) in enumerate(sampled_characters):
			deg = random.sample(range(4), 1)[0]
			_imgs = self.data[char]
			_ind = random.sample(range(len(_imgs)), nb_samples_per_class+nb_samples_per_class_test)
			_ind_tr = _ind[:nb_samples_per_class]
			labels_and_images_support.extend([(k, rotate_right(_imgs[i], deg).flatten()) for i in _ind_tr])
			_ind_test = _ind[nb_samples_per_class:]
			labels_and_images_test.extend([(k, rotate_right(_imgs[i], deg).flatten()) for i in _ind_test])

			
		random.shuffle(labels_and_images_support)
		labels_tr, images_tr = zip(*labels_and_images_support)

		random.shuffle(labels_and_images_test)
		labels_test, images_test = zip(*labels_and_images_test)
		return images_tr, labels_tr, images_test, labels_test


class MnistGenerator(object):
	"""MnistGenerator
	"""
	def __init__(self, test_only=True, lbl_set=range(10), augment=False, nb_classes=5, nb_samples_per_class=10, 
				 batchsize=64, max_iter=None, xp=np, nb_samples_per_class_test=2):
		super(MnistGenerator, self).__init__()
		self.nb_classes = nb_classes
		self.nb_samples_per_class = nb_samples_per_class
		self.nb_samples_per_class_test = nb_samples_per_class_test
		self.batchsize = batchsize
		self.max_iter = max_iter
		self.xp = xp
		self.num_iter = 0
		self.lbl_set = lbl_set
		self.test_only = test_only
		self.augment = augment
		self.data = self._load_data(self.xp)

	def _load_data(self, xp):
		train_all, test_all = chainer.datasets.get_mnist()

		data_dict = {}
		for x,y in test_all:
			data_dict[y] = data_dict.get(y, []) + [x.reshape((28, 28))]

		if not self.test_only:
			for x,y in train_all:
				data_dict[y] = data_dict.get(y, []) + [x.reshape((28, 28))]

		return {key: np.array(val) for (key, val) in data_dict.items()}

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def next(self):
		if (self.max_iter is None) or (self.num_iter < self.max_iter):
			labels_and_images_support = []
			labels_and_images_test = []

			self.num_iter += 1
			sampled_characters = random.sample(self.lbl_set, self.nb_classes) # list of keys

			for _ in xrange(self.batchsize):
				support_set = []

				for k,char in enumerate(sampled_characters):
					deg = random.sample(range(4), 1)[0]
					_imgs = self.data[char]
					_ind = random.sample(range(len(_imgs)), self.nb_samples_per_class+self.nb_samples_per_class_test)
					_ind_tr = _ind[:self.nb_samples_per_class]
					if self.augment:
						support_set.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_tr])
					else:
						support_set.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_tr])
					_ind_test = _ind[self.nb_samples_per_class:]
					if self.augment:
						labels_and_images_test.extend([(k, self.xp.array(rotate_right(_imgs[i], deg).flatten())) for i in _ind_test])
					else:
						labels_and_images_test.extend([(k, self.xp.array(_imgs[i].flatten())) for i in _ind_test])
				random.shuffle(support_set)
				labels_tr, images_tr = zip(*support_set)

				labels_and_images_support.append((images_tr, labels_tr))

			_images, _labels = zip(*labels_and_images_support)
			images_tr = [
				self.xp.concatenate(map(lambda x: x.reshape((1,-1)), _img), axis=0)
				for _img in zip(*_images)]
			labels_tr = [_lbl for _lbl in zip(*_labels)]
			
			random.shuffle(labels_and_images_test)
			labels_test, images_test = zip(*labels_and_images_test)
			images_test = self.xp.concatenate([img.reshape((1, -1)) for img in images_test], axis=0)

			return (self.num_iter - 1), (images_tr, labels_tr, images_test, labels_test)
		else:
			raise StopIteration()


class MiniImageNetGenerator(object):
	"""OmniglotGenerator

	"""
	def __init__(self, data_file, augment=False, mean=None, mean_norm=False,
				 nb_classes=5, nb_samples_per_class=10, 
				 batchsize=64, max_iter=None, xp=np, nb_samples_per_class_test=2):
		super(MiniImageNetGenerator, self).__init__()
		self.data_file = data_file
		self.nb_classes = nb_classes
		self.nb_samples_per_class = nb_samples_per_class
		self.nb_samples_per_class_test = nb_samples_per_class_test
		self.batchsize = batchsize
		self.max_iter = max_iter
		self.xp = xp
		self.num_iter = 0
		self.augment = augment
		self.mean = mean
		self.mean_norm = mean_norm
		self.data = self._load_data(self.data_file, self.xp)
		

	def _load_data(self, data_file, xp):
		data_dict = np.load(data_file)
		if self.mean_norm:
			if self.mean is None:
				self.mean = np.concatenate([val for (key, val) in data_dict.items()], axis=0).mean(axis=0)
			return {key: val-self.mean for (key, val) in data_dict.items()}
		return {key: val for (key, val) in data_dict.items()}

	def __iter__(self):
		return self

	def __next__(self):
		return self.next()

	def next(self):
		if (self.max_iter is None) or (self.num_iter < self.max_iter):
			labels_and_images_support = []
			labels_and_images_test = []

			self.num_iter += 1
			sampled_characters = random.sample(self.data.keys(), self.nb_classes) # list of keys

			for _ in xrange(self.batchsize):
				support_set = []

				for k,char in enumerate(sampled_characters):
					deg = random.sample(range(4), 1)[0]
					_imgs = self.data[char]
					_ind = random.sample(range(len(_imgs)), self.nb_samples_per_class+self.nb_samples_per_class_test)
					_ind_tr = _ind[:self.nb_samples_per_class]
					if self.augment:
						support_set.extend([(k, self.xp.array(rotate_right(_imgs[i], deg))) for i in _ind_tr])
					else:
						support_set.extend([(k, self.xp.array(_imgs[i])) for i in _ind_tr])
					_ind_test = _ind[self.nb_samples_per_class:]
					if self.augment:
						labels_and_images_test.extend([(k, self.xp.array(rotate_right(_imgs[i], deg))) for i in _ind_test])
					else:
						labels_and_images_test.extend([(k, self.xp.array(_imgs[i])) for i in _ind_test])

				random.shuffle(support_set)
				labels_tr, images_tr = zip(*support_set)

				labels_and_images_support.append((images_tr, labels_tr))

			_images, _labels = zip(*labels_and_images_support)
			images_tr = [
				# NOTE: be careful here!
				# I had to permute the dimensions due the output image format from preporcessing
				# please make sure your img dimensions are correct
				self.xp.concatenate(map(lambda x: np.transpose(x, (2, 0, 1)).reshape((1, 3, 84, 84)), _img), axis=0)
				for _img in zip(*_images)]
			labels_tr = [_lbl for _lbl in zip(*_labels)]
			
			random.shuffle(labels_and_images_test)
			labels_test, images_test = zip(*labels_and_images_test)
			# NOTE: be careful here!
			# I had to permute the dimensions due the output image format from preporcessing
			# please make sure your img dimensions are correct
			images_test = self.xp.concatenate([np.transpose(img, (2, 0, 1)).reshape((1, 3, 84, 84)) for img in images_test], axis=0)

			return (self.num_iter - 1), (images_tr, labels_tr, images_test, labels_test)
		else:
			raise StopIteration()