"""
Only last few layers are fast-parameterized and layer-augmented, so with prefix Partial.
The base learner is a separate CNN.

"""
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable, optimizers
import numpy as np
import copy
from utilsf import cosine_similarity2d, logAndSign, logAndSign_gpu, Block, BlockBN

class MetaNetPartial(chainer.Chain):

	def __init__(self, n_out, n_out_test, gpu):
		super(MetaNetPartial, self).__init__(
			block1_1=BlockBN(32, 3),
			block1_2=BlockBN(32, 3),
			block1_3=BlockBN(32, 3),
			block1_4=Block(32, 3),
			block1_5=Block(32, 3),
			fc1=L.Linear(288, 288, nobias=True),
			fc2=L.Linear(288, n_out, nobias=True),
			fc3=L.Linear(288, n_out_test, nobias=True),
			
			key_1=BlockBN(32, 3),
			key_2=BlockBN(32, 3),
			key_3=BlockBN(32, 3),
			key_4=Block(32, 3),
			key_5=Block(32, 3),
			key_fc1=L.Linear(288, 288, nobias=True),
			key_fc2=L.Linear(288, n_out, nobias=True),
			key_fc3=L.Linear(288, n_out_test, nobias=True),

			meta_lstm_l1 = L.LSTM(2, 20),
			meta_lstm_ll1 = L.Linear(20, 1),
			meta_g_lstm_l1 = L.LSTM(2, 20),
			meta_g_lstm_ll1 = L.Linear(20, 1),

			m_l1=L.Linear(2, 20, nobias=False),
			m_ll1=L.Linear(20, 20, nobias=False),
			mc_l1=L.Linear(2, 20, nobias=False),
			mc_ll1=L.Linear(20, 20, nobias=False),
			meta_lstm_l2 = L.Linear(20, 1),
			meta_g_lstm_l2 = L.Linear(20, 1),
		)
		self.__n_out = n_out
		self.__n_out_test = n_out_test
		self.__gpu = gpu
		self.__mod = cuda.cupy if gpu >= 0 else np
		if gpu >= 0:
			cuda.get_device(gpu).use()
			self.to_gpu()

	def init_optimizer(self):
		self.__opt = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)
		self.__opt.setup(self)
		self.__opt.add_hook(chainer.optimizer.GradientClipping(10))
		
	def embed_key(self, train, support_sets, support_lbls, x_set):
		mod = self.__mod
		model = self
		IT = 10
		N = len(support_sets)

		model.meta_lstm_l1.reset_state()
		model.meta_g_lstm_l1.reset_state()

		x = mod.concatenate(support_sets, axis=0).reshape((-1, 3, 84, 84))
		x = Variable(x, volatile=False)
		x = F.dropout(x, ratio=0.0, train=train)
		h = model.key_1(x, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.key_2(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.key_3(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		
		hs = F.split_axis(h, N, axis=0)

		
		inc = 1 if N < 10 else N/IT
		for i in xrange(0, N, inc):
			self.cleargrads()

			h = F.concat(hs[i:(i+inc)], axis=0)

			h = model.key_4(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_5(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.reshape(h, (-1, 288))
			h = F.dropout(h, ratio=0.0, train=train)
			h = F.relu(model.key_fc1(h))
			if train or self.__n_out==self.__n_out_test:
				y = model.key_fc2(h)
			else:
				y = model.key_fc3(h)

			y_batch = mod.array(support_lbls[i:(i+inc)], dtype=np.int32).reshape((-1,))
			lbl = Variable(y_batch, volatile=False)
			loss = F.softmax_cross_entropy(y, lbl)
			loss.backward(retain_grad=True)

			grads = []
			grad_sections = []
			grads.append(model.key_4.conv.W.grad.reshape(-1, 1))
			grad_sections.append(grads[-1].shape[0])

			
			grads.append(model.key_5.conv.W.grad.reshape(-1, 1))

			grads1 = []
			grad_sections1 = []
			grads1.append(model.key_fc1.W.grad.reshape(-1, 1))
			grad_sections1.append(grads1[-1].shape[0])

			if train or self.__n_out==self.__n_out_test:
				grads1.append(model.key_fc2.W.grad.reshape(-1, 1))
			else:
				grads1.append(model.key_fc3.W.grad.reshape(-1, 1))
			
			meta_in = mod.concatenate(grads, axis=0)
			meta_in = cuda.to_cpu(meta_in)
			meta_in = logAndSign(meta_in, k=7)
			meta_in = mod.array(meta_in)
			meta_in = Variable(meta_in, volatile=False)
			
			meta_outs = model.meta_lstm_l1(F.dropout(meta_in, ratio=0.0, train=train))
			meta_outs = model.meta_lstm_ll1(F.dropout(meta_outs, ratio=0.0, train=train))
		 
			meta_in = mod.concatenate(grads1, axis=0)
			meta_in = cuda.to_cpu(meta_in)
			meta_in = logAndSign(meta_in, k=7)
			meta_in = mod.array(meta_in)	
			meta_in = Variable(meta_in, volatile=False)

			meta_outs1 = model.meta_g_lstm_l1(F.dropout(meta_in, ratio=0.0, train=train))
			meta_outs1 = model.meta_g_lstm_ll1(F.dropout(meta_outs1, ratio=0.0, train=train))

		meta_outs = F.split_axis(meta_outs, grad_sections, axis=0)
		meta_outs1 = F.split_axis(meta_outs1, grad_sections1, axis=0)

		key_4_W = F.reshape(meta_outs[0], model.key_4.conv.W.data.shape)
		key_5_W = F.reshape(meta_outs[1], model.key_5.conv.W.data.shape)
		key_fc1_W = F.reshape(meta_outs1[0], model.key_fc1.W.data.shape)

		self.cleargrads()

		keys = []
		for x in [support_sets, x_set]:
			x = mod.asarray(x, dtype=np.float32).reshape((-1, 3, 84, 84))
			x = Variable(x, volatile=False)
			
			x = F.dropout(x, ratio=0.0, train=train)
			h = model.key_1(x, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_2(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_3(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_4(h, train) + model.key_4.call_on_W(h, key_4_W, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_5(h, train) + model.key_5.call_on_W(h, key_5_W, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.reshape(h, (-1, 288))
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.key_fc1(h) + F.matmul(h, key_fc1_W, transb=True)
			keys.append(h)

		return keys

	def __forward(self, train, support_sets, support_lbls, x_set, x_lbl = None):
		model = self
		mod = self.__mod
		gpu = self.__gpu
		n_out = self.__n_out
		batch_size = support_sets[0].shape[0]
		N = len(support_sets)
		
		grad_mems = []
		grad_mems1 = []
		key_mems = []

		x = mod.concatenate(support_sets, axis=0).reshape((-1, 3, 84, 84))
		x = Variable(x, volatile=False)
		x = F.dropout(x, ratio=0.0, train=train)

		h = model.block1_1(x, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.block1_2(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.block1_3(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		
		hs = F.split_axis(h, N, axis=0)

		for i in range(N):
			self.cleargrads()
			
			h = hs[i]

			h = model.block1_4(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.block1_5(h, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.reshape(h, (1, 288))
			h = F.dropout(h, ratio=0.0, train=train)
			h = F.relu(model.fc1(h))
			h = F.dropout(h, ratio=0.0, train=train)
			if train or self.__n_out==self.__n_out_test:
				y = model.fc2(h)
			else:
				y = model.fc3(h)
			
			y_batch = mod.array(support_lbls[i], dtype=np.int32)
			lbl = Variable(y_batch, volatile=False)
			support_loss = F.softmax_cross_entropy(y, lbl)

			support_loss.backward(retain_grad=True)
			
			grads = []
			grad_sections = []
			grads.append(model.block1_4.conv.W.grad.reshape(-1, 1))
			grad_sections.append(grads[-1].shape[0])

			
			grads.append(model.block1_5.conv.W.grad.reshape(-1, 1))

			grads1 = []
			grad_sections1 = []
			grads1.append(model.fc1.W.grad.reshape(-1, 1))
			grad_sections1.append(grads1[-1].shape[0])

			if train or self.__n_out==self.__n_out_test:
				grads1.append(model.fc2.W.grad.reshape(-1, 1))
			else:
				grads1.append(model.fc3.W.grad.reshape(-1, 1))
			
			meta_in = mod.concatenate(grads, axis=0)
			meta_in = cuda.to_cpu(meta_in)
			meta_in = logAndSign(meta_in, k=7)
			meta_in = mod.array(meta_in)
			meta_in = Variable(meta_in, volatile=False)
			
			
			meta_outs = F.relu(model.m_l1(F.dropout(meta_in, ratio=0.0, train=train)))
			meta_outs = F.relu(model.m_ll1(F.dropout(meta_outs, ratio=0.0, train=train)))
			meta_outs = model.meta_lstm_l2(F.dropout(meta_outs, ratio=0.0, train=train))
			
			grad_mems.append(meta_outs)

			meta_in = mod.concatenate(grads1, axis=0)
			meta_in = cuda.to_cpu(meta_in)
			meta_in = logAndSign(meta_in, k=7)
			meta_in = mod.array(meta_in)
			
			meta_in = Variable(meta_in, volatile=False)
			
			
			meta_outs = F.relu(model.mc_l1(F.dropout(meta_in, ratio=0.0, train=train)))
			meta_outs = F.relu(model.mc_ll1(F.dropout(meta_outs, ratio=0.0, train=train)))
			meta_outs = model.meta_g_lstm_l2(F.dropout(meta_outs, ratio=0.0, train=train))
			grad_mems1.append(meta_outs)

		grad_mems = F.concat(grad_mems, axis=1)
		grad_mems1 = F.concat(grad_mems1, axis=1)

		self.cleargrads()
		key_mems, x_keys = self.embed_key(train, support_sets, support_lbls, x_set)

		x_set = x_set.reshape((-1, 3, 84, 84))

		x_keys = F.split_axis(x_keys, x_set.shape[0], axis=0)

		x = Variable(x_set, volatile=False)

		x = F.dropout(x, ratio=0.0, train=train)
		h = model.block1_1(x, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.block1_2(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		h = model.block1_3(h, train)
		h = F.max_pooling_2d(h, ksize=2, stride=2)
		h = F.dropout(h, ratio=0.0, train=train)
		

		xs = F.split_axis(h, x_set.shape[0], axis=0)

		x_loss = 0
		preds = []
		for h, x_key, lbl in zip(xs, x_keys, x_lbl):
			x_key = F.reshape(x_key, (1, -1))
			sc = F.softmax(cosine_similarity2d(key_mems, x_key))
			meta_outs = F.matmul(grad_mems, sc, transb=True)
			meta_outs1 = F.matmul(grad_mems1, sc, transb=True)

			meta_outs = F.split_axis(meta_outs, grad_sections, axis=0)
			meta_outs1 = F.split_axis(meta_outs1, grad_sections1, axis=0)

			block1_4_W = F.reshape(meta_outs[0], model.block1_4.conv.W.data.shape)
			block1_5_W = F.reshape(meta_outs[1], model.block1_5.conv.W.data.shape)
			fc1_W = F.reshape(meta_outs1[0], model.fc1.W.data.shape)
			if train or self.__n_out==self.__n_out_test:
				fc2_W = F.reshape(meta_outs1[1], model.fc2.W.data.shape)
			else:
				fc2_W = F.reshape(meta_outs1[1], model.fc3.W.data.shape)
			
			h = model.block1_4(h, train) + model.block1_4.call_on_W(h, block1_4_W, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.dropout(h, ratio=0.0, train=train)
			h = model.block1_5(h, train) + model.block1_5.call_on_W(h, block1_5_W, train)
			h = F.max_pooling_2d(h, ksize=2, stride=2)
			h = F.reshape(h, (1, 288))
			h = F.dropout(h, ratio=0.0, train=train)
			h = F.relu(model.fc1(h)) + F.relu(F.matmul(h, fc1_W, transb=True))
			h = F.dropout(h, ratio=0.0, train=train)
			if train or self.__n_out==self.__n_out_test:
				y = model.fc2(h) + F.matmul(h, fc2_W, transb=True)
			else:
				y = model.fc3(h) + F.matmul(h, fc2_W, transb=True)
			
			y_batch = mod.array(lbl, dtype=np.int32).reshape((1,))
			lbl = Variable(y_batch, volatile=False)
			x_loss += F.softmax_cross_entropy(y, lbl)
			preds += mod.argmax(y.data, 1).tolist()

		return preds, x_loss

	def train(self, support_sets, support_lbls, x_set, x_lbl):
		preds, x_loss = self.__forward(True, support_sets, support_lbls, x_set, x_lbl=x_lbl)
		self.cleargrads()
		x_loss.backward()
		self.__opt.update()
		return preds, x_loss

	def predict(self, support_sets, support_lbls, x_set, x_lbl):
		self.cleargrads()
		preds, x_loss = self.__forward(False, support_sets, support_lbls, x_set, x_lbl)
		return preds, x_loss