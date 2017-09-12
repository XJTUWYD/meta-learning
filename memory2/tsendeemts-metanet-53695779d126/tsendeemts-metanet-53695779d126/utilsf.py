"""
Some utility functions for MetaNet

"""
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable, optimizers
import numpy as np

def cosine_similarity2d(x, y, eps=1e-6):
	n1, n2 = x.data.shape
	_, m2 = y.data.shape
	z = F.reshape(F.matmul(x, y, transb=True),  (1, -1))
	x2 = F.reshape(F.sum(x * x, axis=1), (1, n1))
	y2 = F.broadcast_to(F.reshape(F.sum(y * y, axis=1), (1, 1)), (1, n1))
	z /= F.exp(F.log(x2 * y2 + eps) / 2)
	return z

def clamp(inputs, min_value=None, max_value=None):
	output = inputs
	if min_value is not None:
		output[output < min_value] = min_value
	if max_value is not None:
		output[output > max_value] = max_value
	return output

def clamp_gpu(inputs, mod, min_value=None, max_value=None):
	output = inputs.copy()
	if min_value is not None:
		mod.copyto(output, output.dtype.type(min_value), where=output<min_value)
	if max_value is not None:
		mod.copyto(output, output.dtype.type(max_value), where=output>max_value)
	return output

def logAndSign(inputs, k=5):
	eps = np.finfo(inputs.dtype).eps
	log = np.log(np.absolute(inputs) + eps)
	clamped_log = clamp(log / k, min_value=-1.0)
	sign = clamp(inputs * np.exp(k), min_value=-1.0, max_value=1.0)
	return np.concatenate([clamped_log, sign], axis=1)

def logAndSign_gpu(inputs, mod, k=5):
	eps = np.finfo(inputs.dtype).eps
	log = mod.log(mod.absolute(inputs) + eps)
	clamped_log = clamp_gpu(log / k, mod, min_value=-1.0)
	sign = clamp_gpu(inputs * np.exp(k), mod, min_value=-1.0, max_value=1.0)
	return mod.concatenate([clamped_log, sign], axis=1)

class Block(chainer.Chain):

	"""A convolution, batch norm, ReLU block.
	A block in a feedforward network that performs a
	convolution followed by batch normalization followed
	by a ReLU activation.
	For the convolution operation, a square filter size is used.
	Args:
		out_channels (int): The number of output channels.
		ksize (int): The size of the filter is ksize x ksize.
		pad (int): The padding to use for the convolution.
	"""

	def __init__(self, out_channels, ksize, pad=1):
		super(Block, self).__init__(
			conv=L.Convolution2D(None, out_channels, ksize, pad=pad,
								 nobias=True)
		)
		self.pad = pad

	def __call__(self, x, train=True):
		h = self.conv(x)
		return F.relu(h)

	def call_on_W(self, x, W, train=True):
		h = F.convolution_2d(x, W, pad=self.pad)
		return F.relu(h)

class BlockBN(chainer.Chain):

	"""A convolution, batch norm, ReLU block.
	A block in a feedforward network that performs a
	convolution followed by batch normalization followed
	by a ReLU activation.
	For the convolution operation, a square filter size is used.
	Args:
		out_channels (int): The number of output channels.
		ksize (int): The size of the filter is ksize x ksize.
		pad (int): The padding to use for the convolution.
	"""

	def __init__(self, out_channels, ksize, pad=1):
		super(BlockBN, self).__init__(
			conv=L.Convolution2D(None, out_channels, ksize, pad=pad,
								 nobias=True)
			,bn=L.BatchNormalization(out_channels)
		)
		self.pad = pad

	def __call__(self, x, train=True):
		h = self.conv(x)
		h = self.bn(h, test=not train)
		return F.relu(h)

	def call_on_W(self, x, W, train=True):
		h = F.convolution_2d(x, W, pad=self.pad)
		h = self.bn(h, test=not train)
		return F.relu(h)
