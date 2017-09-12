"""
This is a script used to download and resize the imagenet subset
To use the exact images used in MetaNet experiments, modify the function resize_save.

"""

import math
import os
import time
import copy
import pandas as pd
import numpy as np
import six
from collections import Counter
import gc
import argparse
import  tarfile
from PIL import Image
import scipy.misc
from collections import defaultdict
import warnings
warnings.filterwarnings("error")

#class split from https://github.com/twitter/meta-learning-lstm
input_dir = 'path/to/meta-learning-lstm/data/miniImagenet/'
#paht to imagenet subset
data_dir = 'path/to/imagenet/imagenet/miniimagenet/'

def _read_image_as_array(path, dtype='int32'):
	f = Image.open(path)
	try:
		image = np.asarray(f, dtype=dtype)
	finally:
		# Only pillow >= 3.0 has 'close' method
		if hasattr(f, 'close'):
			f.close()
	return image

def download_imgs():
	train = pd.read_csv(input_dir + '/train.csv', sep=',')
	test = pd.read_csv(input_dir + '/test.csv', sep=',')
	dev = pd.read_csv(input_dir + '/val.csv', sep=',')
	labels = train.label.unique().tolist() + test.label.unique().tolist() + dev.label.unique().tolist()
	for lbl in labels:
		os.system('wget "http://image-net.org/download/synset?wnid=' + lbl + '&username=username&accesskey=accesskey&release=latest&src=stanford" -O' + lbl + '.tar')

# download_imgs()

#modify this script to use the exact images used in MetaNet experiments
def resize_save(data_path, out_file):
	dataset = pd.read_csv(data_path, sep=',')
	lbls = dataset.label.unique().tolist()
	tmp_dict = defaultdict(list)
	file_names = []
	lbl_names = []
	for lbl in lbls:
		print lbl
		tar = tarfile.open(data_dir + lbl + ".tar")
		imgs = tar.getmembers()
		np.random.shuffle(imgs)
		# imgs = imgs[:600]
		c = 0
		for img in imgs:
			f = tar.extractfile(img)
			try:
				img_array = _read_image_as_array(f)
				img_array = scipy.misc.imresize(img_array, (84, 84))
				img_array = img_array.astype('float32')
				img_array *= (1.0 / 255.0)
				tmp_dict[lbl].append(img_array.reshape((1, 84, 84, 3)))
				file_names.append(img.name)
				lbl_names.append(lbl)
				c += 1
			# Reading/resizing some image throws exception, so had to filter them out. 
			except Exception, e:
				print "skipping image..."
			if c == 600:
				break
	results = {key : np.concatenate(value) for key, value in tmp_dict.items()}
	np.savez(data_dir + out_file + ".npz", **results)
	sub = pd.DataFrame({'img_file': file_names, 'label': lbl_names})
	sub.to_csv(data_dir + out_file + '.csv', index=False)

print "train..."
resize_save(input_dir + '/train.csv', 'train')
print "test..."
resize_save(input_dir + '/test.csv', 'test')
print "dev..."
resize_save(input_dir + '/val.csv', 'dev')