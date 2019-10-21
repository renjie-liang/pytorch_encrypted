"""Native tensors and their factory.

These use TensorFlow's native dtypes tf.int32 and tf.int64 for the given float
encoding being used (fixed-point, etc.)."""
from __future__ import absolute_import
from typing import Union, List, Dict, Tuple, Optional
import abc
import math

import numpy as np
import torch
import tensorflow as tf

from .factory import (AbstractFactory, AbstractTensor, AbstractVariable,
											AbstractConstant, AbstractPlaceholder)
# from .helpers import inverse
from .shared import binarize, im2col ,conv2d
from .shared import im2col
# from ..operations import secure_random


def native_factory(NATIVE_TYPE, EXPLICIT_MODULUS=None):	# pylint: disable=invalid-name
	"""Constructs the native tensor Factory."""

	class Factory(AbstractFactory):
		"""Native tensor factory."""

		def tensor(self, value):

			if isinstance(value, torch.Tensor):
				if value.dtype is not NATIVE_TYPE:
					value = value.to(NATIVE_TYPE)
				return DenseTensor(value)

			if isinstance(value, np.ndarray):
				value = torch.from_numpy(value)
				value = value.to(self.native_type)

				return DenseTensor(value)

			raise TypeError("Don't know how to handle {}".format(type(value)))

		# def from_numpy(self, value):
		# 	if isinstance(value, np.ndarray):
		# 		value = torch.from_numpy(value, dtype=self.native_type)
		# 		return DenseTensor(value) ###

		# 	raise TypeError("Don't know how to handle {}".format(type(value)))

		def constant(self, value):
			if isinstance(value, np.ndarray):
				value = torch.from_numpy(value)
				value = value.to(self.native_type)
				return DenseTensor(value) ###

			raise TypeError("Don't know how to handle {}".format(type(value)))

		# def variable(self, initial_value):

		# 	if isinstance(initial_value, (torch.Tensor, np.ndarray)):
		# 		return Variable(initial_value)

		# 	if isinstance(initial_value, Tensor):
		# 		return Variable(initial_value.value)

		# 	msg = "Don't know how to handle {}"
		# 	raise TypeError(msg.format(type(initial_value)))

		# def placeholder(self, shape):
		# 	return Placeholder(shape)

		@property
		def min(self):
			if EXPLICIT_MODULUS is not None:
				return 0
			if NATIVE_TYPE is torch.int32:
				return -(2 ** 31)
			elif NATIVE_TYPE is torch.int64:
				return -(2 ** 63)


		@property
		def max(self):
			if EXPLICIT_MODULUS is not None:
				return EXPLICIT_MODULUS

			if NATIVE_TYPE is torch.int32:
				return (2 ** 31) - 1 
			elif NATIVE_TYPE is torch.int64:
				return  (2 ** 63) - 1 




		@property
		def modulus(self) -> int:
			if EXPLICIT_MODULUS is not None:
				return EXPLICIT_MODULUS
			if NATIVE_TYPE is torch.int32:
				return 2 ** 32
			elif NATIVE_TYPE is torch.int64:
				return 2 ** 64

		@property
		def native_type(self):
			return NATIVE_TYPE

		def sample_uniform(self, shape, minval: Optional[int] = None, maxval: Optional[int] = None):
			minval = minval or self.min
			# TODO(Morten) believe this should be native_type.max+1
			maxval = maxval or self.max

			value = torch.randint(low=minval , high = maxval, size = shape, dtype=NATIVE_TYPE)

			return DenseTensor(value)


		def sample_uniform_mask(self, shape, minval: Optional[int] = None,  maxval: Optional[int] = None):
			pass

		def sample_bounded(self, shape, bitlength: int):
			maxval = 2 ** bitlength
			assert maxval <= self.max

			# if secure_random.supports_seeded_randomness():
			# 	seed = secure_random.secure_seed()
			# 	return UniformTensor(shape=shape, seed=seed, minval=0, maxval=maxval)

			# if secure_random.supports_secure_randomness():
			# 	sampler = secure_random.random_uniform
			# else:
			# 	sampler = tf.random_uniform
			sampler = torch.randint
			value = sampler(low=0 , high = maxval, size = shape, dtype=NATIVE_TYPE)
			return DenseTensor(value)

		def sample_bits(self, shape):
			return self.sample_bounded(shape, bitlength=1)

		# def stack(self, xs: list, axis: int = 0):
		# 	assert all(isinstance(x, Tensor) for x in xs)
		# 	value = tf.stack([x.value for x in xs], axis=axis)
		# 	return DenseTensor(value)

		# def concat(self, xs: list, axis: int):
		# 	assert all(isinstance(x, Tensor) for x in xs)
		# 	value = tf.concat([x.value for x in xs], axis=axis)
		# 	return DenseTensor(value)

	FACTORY = Factory()	# pylint: disable=invalid-name

	def _lift(x, y) -> Tuple['Tensor', 'Tensor']:

		if isinstance(x, Tensor) and isinstance(y, Tensor):
			return x, y

		if isinstance(x, Tensor):

			if isinstance(y, int):
				return x, x.factory.tensor(np.array([y]))

		if isinstance(y, Tensor):

			if isinstance(x, int):
				return y.factory.tensor(np.array([x])), y

		raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

	class Tensor(AbstractTensor):
		"""Base class for other native tensor classes."""

		@property
		@abc.abstractproperty
		def value(self):
			pass

		@property
		@abc.abstractproperty
		def shape(self):
			pass

		def identity(self):
			value = self.value.clone()
			return DenseTensor(value)

		def clone(self):
			value = self.value.clone()
			return DenseTensor(value)

		def to_native(self) -> torch.Tensor:
			return self.value

		def bits(self, factory=None) -> AbstractTensor:
			factory = factory or FACTORY
			if EXPLICIT_MODULUS is None:
				res =  factory.tensor(binarize(self.value))
				return res


			bitsize = bitsize = math.ceil(math.log2(EXPLICIT_MODULUS))
			return factory.tensor(binarize(self.value % EXPLICIT_MODULUS, bitsize))

		def __repr__(self) -> str:
			return '{}(shape={})'.format(type(self), self.shape)

		@property
		def factory(self):
			return FACTORY

		def __add__(self, other):
			x, y = _lift(self, other)
			return x.add(y)

		def __radd__(self, other):
			x, y = _lift(self, other)
			return y.add(x)

		def __sub__(self, other):
			x, y = _lift(self, other)
			return x.sub(y)

		def __rsub__(self, other):
			x, y = _lift(self, other)
			return y.sub(x)

		def __mul__(self, other):
			x, y = _lift(self, other)
			return x.mul(y)

		def __rmul__(self, other):
			x, y = _lift(self, other)
			return x.mul(y)

		def __mod__(self, k: int):
			return self.mod(k)

		def __neg__(self):
			return self.mul(-1)

		def __getitem__(self, slc):
			return DenseTensor(self.value[slc])

		def add(self, other):
			x, y = _lift(self, other)
			value = x.value + y.value
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)

		def sub(self, other):
			x, y = _lift(self, other)
			value = x.value - y.value
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)

		def mul(self, other):
			x, y = _lift(self, other)
			value = x.value * y.value
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)


		def matmul(self, other):
			x, y = _lift(self, other)
			value = torch.mm(x.value, y.value)
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)

		def mm(self, other):
			return self.matmul(other)

		def im2col(self, h_filter, w_filter, padding, stride):
			i2c = im2col(self.value, h_filter, w_filter, padding, stride)
			return DenseTensor(i2c)

		# def conv2d(self, other, stride: int, padding: str = 'SAME'):
		# 	if EXPLICIT_MODULUS is not None:
		# 		# TODO(Morten) any good reason this wasn't implemented for PrimeTensor?
		# 		raise NotImplementedError()
		# 	x, y = _lift(self, other)
		# 	return conv2d(x, y, stride, padding)

		# def batch_to_space_nd(self, block_shape, crops):
		# 	value = tf.batch_to_space_nd(self.value, block_shape, crops)
		# 	return DenseTensor(value)

		# def space_to_batch_nd(self, block_shape, paddings):
		# 	value = tf.space_to_batch_nd(self.value, block_shape, paddings)
		# 	return DenseTensor(value)

		def mod(self, k: int):
			value = self.value % k
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)

		def transpose(self, perm):
			res = self.value.permute(*perm)
			return DenseTensor(res)

		def permute(self, perm):
			res = self.value.permute(*perm)
			return DenseTensor(res)


		# def strided_slice(self, args, kwargs):
		# 	return DenseTensor(tf.strided_slice(self.value, *args, **kwargs))

		# def gather(self, indices: list, axis: int = 0):
		# 	return DenseTensor(tf.gather(self.value, indices, axis=axis))

		def split(self, num_split: Union[int, list], axis: int = 0):
			values = torch.split(self.value, num_split, dim=axis)
			return [DenseTensor(value) for value in values]

		def reshape(self, axes: Union[tf.Tensor, List[int]]):

			res = self.value.reshape(axes)

			return DenseTensor(res)

		def negative(self):
			value = self.value * -1
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)
		def sum(self,axis = None):
			value = torch.sum(self.value,dim = axis)
			return DenseTensor(value)

		def reduce_sum(self, axis):
			value = torch.sum(self.value,dim = axis)
			if EXPLICIT_MODULUS is not None:
				value %= EXPLICIT_MODULUS
			return DenseTensor(value)

		def cumsum(self, axis, exclusive, reverse):
			
			if exclusive is True and reverse is True:
				pad = torch.zeros_like(self.value).sum(dim = axis)
				pad.unsqueeze_(dim = axis)
				temp = torch.cat([self.value, pad], dim = axis)
				temp  = torch.flip(temp ,dims = [axis])
				temp = torch.cumsum(temp, dim = axis)
				temp  = torch.flip(temp ,dims = [axis])
				split_size = temp.size(-1) -1
				temp = torch.split(temp,split_size_or_sections =split_size, dim = -1)
				temp = temp[0]

				return  DenseTensor(temp)
			else:
				raise NotImplementedError

			# if EXPLICIT_MODULUS is not None:
			# 	value %= EXPLICIT_MODULUS
			# return DenseTensor(value)

		def equal_zero(self, factory=None):
			factory = factory or FACTORY
			res = (self.value == 0)
			res.to(factory.native_type)
			return factory.tensor(res)

		def equal(self, other, factory=None):
			x, y = _lift(self, other)
			factory = factory or FACTORY
			return factory.tensor(x.value == y.value)

		def truncate(self, amount, base=2):

			if base == 2:
				return self.right_shift(amount)

			factor = base**amount
			factor_inverse = inverse(factor, self.factory.modulus)
			return (self - (self % factor)) * factor_inverse

		def right_shift(self, bitlength):
			x_np = self.value.numpy()
			x_np = np.right_shift(x_np, bitlength)
			x = torch.from_numpy(x_np) 

			return DenseTensor(x)

		def expand_dims(self, axis: Optional[int] = None):
			res = self.value.unsqueeze(axis)
			return DenseTensor(res)

		def squeeze(self, axis: Optional[List[int]] = None):
			return DenseTensor(torch.squeeze(self.value, dim=axis))

		def cast(self, factory):
			return factory.tensor(self.value)

	class DenseTensor(Tensor):
		"""Public native Tensor class."""

		def __init__(self, value):
			self._value = value

		@property
		def shape(self):
			return self._value.shape

		@property
		def value(self):
			return self._value

		@property
		def support(self):
			return [self._value]

		def to(self, device = None):
			self._value.to(device)
	# class UniformTensor(Tensor):
	# 	"""Class representing a uniform-random, lazily sampled tensor.

	# 	Lazy sampling optimizes communication by sending seeds in place of
	# 	fully-expanded tensors."""

	# 	def __init__(self, shape, seed, minval, maxval):
	# 		self._seed = seed
	# 		self._shape = shape
	# 		self._minval = minval
	# 		self._maxval = maxval

	# 	@property
	# 	def shape(self):
	# 		return self._shape

	# 	@property
	# 	def value(self):
	# 		with tf.name_scope('expand-seed'):
	# 			return secure_random.seeded_random_uniform(
	# 					shape=self._shape,
	# 					dtype=NATIVE_TYPE,
	# 					minval=self._minval,
	# 					maxval=self._maxval,
	# 					seed=self._seed,
	# 			)

	# 	@property
	# 	def support(self):
	# 		return [self._seed]

	# class Constant(DenseTensor):
	# 	"""Native Constant class."""

	# 	def __init__(self, constant: tf.Tensor) -> None:
	# 		assert isinstance(constant, tf.Tensor)
	# 		super(Constant, self).__init__(constant)

	# 	def __repr__(self) -> str:
	# 		return 'Constant(shape={})'.format(self.shape)

	# class Placeholder(DenseTensor, AbstractPlaceholder):
	# 	"""Native Placeholder class."""

	# 	def __init__(self, shape: List[int]) -> None:
	# 		self.placeholder = tf.placeholder(NATIVE_TYPE, shape=shape)
	# 		super(Placeholder, self).__init__(self.placeholder)

	# 	def __repr__(self) -> str:
	# 		return 'Placeholder(shape={})'.format(self.shape)

	# 	def feed(self, value: np.ndarray) -> Dict[tf.Tensor, np.ndarray]:
	# 		assert isinstance(value, np.ndarray), type(value)
	# 		return {
	# 				self.placeholder: value
	# 		}

	# class Variable(DenseTensor, AbstractVariable):
	# 	"""Native Variable class."""

	# 	def __init__(self, initial_value: Union[tf.Tensor, np.ndarray]) -> None:
	# 		self.variable = tf.Variable(
	# 				initial_value, dtype=NATIVE_TYPE, trainable=False)
	# 		self.initializer = self.variable.initializer
	# 		super(Variable, self).__init__(self.variable.read_value())

	# 	def __repr__(self) -> str:
	# 		return 'Variable(shape={})'.format(self.shape)

	# 	def assign_from_native(self, value: np.ndarray) -> tf.Operation:
	# 		assert isinstance(value, np.ndarray), type(value)
	# 		return self.assign_from_same(FACTORY.tensor(value))

	# 	def assign_from_same(self, value: Tensor) -> tf.Operation:
	# 		assert isinstance(value, Tensor), type(value)
	# 		return tf.assign(self.variable, value.value).op

	return FACTORY
