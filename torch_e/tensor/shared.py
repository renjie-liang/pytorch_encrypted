"""Commonly used tensor functions."""
import math
from typing import Union, Optional

import torch
import numpy as np

from .factory import AbstractTensor
import tensorflow as tf
def binarize(tensor: torch.Tensor,
						 bitsize: Optional[int] = None) -> tf.Tensor:
	"""Extract bits of values in `tensor`, returning a `tf.Tensor` with same
	dtype."""

	#with tf.name_scope('binarize'):
	if bitsize is None:
		if tensor.dtype == torch.int64:
			bitsize =  64
		elif tensor.dtype == torch.int32:
			bitsize =  32
	np_x = tensor.numpy()

	bitsize = bitsize or (tensor.dtype.size * 8)

	bit_indices_shape = [1] * len(tensor.shape) + [bitsize]

	bit_indices = torch.arange(start = 0, end = bitsize, dtype=tensor.dtype)
	bit_indices = torch.reshape(bit_indices, bit_indices_shape)
	bit_indices_np = bit_indices.numpy()

	val_np = np.expand_dims(np_x, -1)
	val_np = np.bitwise_and(np.right_shift(val_np, bit_indices_np), 1)

	res = torch.from_numpy(val_np)
	assert res.dtype == tensor.dtype
	return res


def bits(tensor: torch.Tensor, bitsize: Optional[int] = None) -> list:
	"""Extract bits of values in `tensor`, returning a list of tensors."""

	if bitsize is None:
		if tensor.dtype == torch.int64:
			bitsize =  64
		elif tensor.dtype == torch.int32:
			bitsize =  32
	np_x = tensor.numpy()

	the_bits = [np.bitwise_and(np.right_shift(np_x, i), 1) 
				 for i in range(bitsize)]
	res = torch.from_numpy(the_bits)
	return res
		# return tf.stack(bits, axis=-1)


def im2col(x: Union[torch.Tensor, np.ndarray],
					 h_filter: int,
					 w_filter: int,
					 padding: str,
					 stride: int) -> torch.Tensor:
	"""Generic implementation of im2col on tf.Tensors."""


		# we need NHWC because tf.extract_image_patches expects this

	if x is torch.Tensor:
		x = x.numpy()
	if x is np.ndarray:
		pass
	temp_tf = tf.constant(x)


	temp_tf = tf.transpose(temp_tf, [0, 2, 3, 1])

	channels =  int(temp_tf.shape[3])

	# extract patches
	patch_tensor = tf.extract_image_patches(
			temp_tf,
			ksizes=[1, h_filter, w_filter, 1],
			strides=[1, stride, stride, 1],
			rates=[1, 1, 1, 1],
			padding=padding
	)

	# change back to NCHW
	patch_tensor_nchw = tf.reshape(tf.transpose(patch_tensor, [3, 1, 2, 0]),
		(h_filter, w_filter, channels, -1))

	# reshape to x_col
	x_col_tensor = tf.reshape(tf.transpose(patch_tensor_nchw, [2, 0, 1, 3]),
												(channels * h_filter * w_filter, -1))
	with tf.compat.v1.Session() as sess:
		res = sess.run(x_col_tensor)

	temp_torch = torch.tensor(res, dtype = torch.int64)
	# print('im2col')
	# print(temp_torch.size())
	# input()
	return  temp_torch

def conv2d(x: AbstractTensor,
					 y: AbstractTensor,
					 stride,
					 padding) -> AbstractTensor:
	"""Generic convolution implementation with im2col over AbstractTensors."""

	with tf.name_scope('conv2d'):

		h_filter, w_filter, in_filters, out_filters = map(int, y.shape)
		n_x, c_x, h_x, w_x = map(int, x.shape)

		if c_x != in_filters:
			# in depthwise conv the filter's in and out dimensions are reversed
			out_filters = in_filters

		if padding == 'SAME':
			h_out = int(math.ceil(float(h_x) / float(stride)))
			w_out = int(math.ceil(float(w_x) / float(stride)))
		elif padding == 'VALID':
			h_out = int(math.ceil(float(h_x - h_filter + 1) / float(stride)))
			w_out = int(math.ceil(float(w_x - w_filter + 1) / float(stride)))
		else:
			raise ValueError("Don't know padding method '{}'".format(padding))

		x_col = x.im2col(h_filter, w_filter, padding, stride)
		w_col = y.transpose([3, 2, 0, 1]).reshape([int(out_filters), -1])
		out = w_col.matmul(x_col)

		out = out.reshape([out_filters, h_out, w_out, n_x])
		out = out.transpose([3, 0, 1, 2])

		return out
