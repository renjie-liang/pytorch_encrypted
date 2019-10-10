# """Odd tensors abstraction. For internal use with SecureNN subprotocols."""
from __future__ import absolute_import
from typing import Tuple, Optional
# from functools import partial

import abc
import math

import torch
import numpy as np

# from ...tensor.factory import AbstractTensor
from factory import AbstractTensor
# from ...tensor.shared import binarize
# from ...operations import secure_random

def odd_factory(NATIVE_TYPE):	# pylint: disable=invalid-name
	"""
	Produces a Factory for OddTensors with underlying tf.dtype NATIVE_TYPE.
	"""

	assert NATIVE_TYPE in (torch.int32, torch.int64)
	class Factory:

		def tensor(self, value):
			if isinstance(value, torch.Tensor):
				if value.dtype is not NATIVE_TYPE:
					value = value.to(NATIVE_TYPE)
				# no assumptions are made about the tensor here and hence we need to
				# apply our mapping for invalid values
				value = _map_minusone_to_zero(value)

				return OddDenseTensor(value)

			raise TypeError("Don't know how to handle {}".format(type(value)))

		def constant(self, value):
			raise NotImplementedError()

		def variable(self, initial_value):
			raise NotImplementedError()

		def placeholder(self, shape):
			raise NotImplementedError()

		@property
		def modulus(self):

			if NATIVE_TYPE is torch.int32:
				return 2**32 - 1

			if NATIVE_TYPE is torch.int64:
				return 2**64 - 1

			raise NotImplementedError(("Incorrect native type ",
																 "{}.".format(NATIVE_TYPE)))

		@property
		def native_type(self):
			return NATIVE_TYPE

		### not prefect
		def sample_uniform(self, shape, minval: Optional[int] = None, maxval: Optional[int] = None):
			"""Sample a tensor from a uniform distribution."""
			assert minval is None
			assert maxval is None

			# if secure_random.supports_seeded_randomness():
			# 	seed = secure_random.secure_seed()
			# 	return OddUniformTensor(shape=shape, seed=seed)

			# if secure_random.supports_secure_randomness():
			# 	sampler = secure_random.random_uniform
			# else:
			# 	sampler = tf.random_uniform
			sampler = torch.randint
			value = _construct_value_from_sampler(sampler=sampler, shape=shape)
			return OddDenseTensor(value)



		def sample_bounded(self, shape, bitlength: int):
			raise NotImplementedError()

		def stack(self, xs: list, axis: int = 0):
			raise NotImplementedError()

		def concat(self, xs: list, axis: int):
			raise NotImplementedError()

	master_factory = Factory()


	class OddTensor(AbstractTensor):
		"""
		Base class for the concrete odd tensors types.

		Implements basic functionality needed by SecureNN subprotocols from a few
		abstract properties implemented by concrete types below.
		"""

		@property
		def factory(self):
			return master_factory

		### these fun like unuse
		@property
		@abc.abstractproperty
		def value(self) -> torch.Tensor:
			pass

		@property
		@abc.abstractproperty
		def shape(self):
			pass

		def identity(self):
			value = self.value.clone()
			return OddDenseTensor(value)

		def clone(self):
			value = self.value.clone()
			return OddDenseTensor(value)


		def __repr__(self) -> str:
			# return '{}(class = {}, size={}, NATIVE_TYPE={})'.format(self.value, type(self),
			# 						self.size(),
			# 						NATIVE_TYPE,
			# )			
			return '{}(class = {})'.format(self.value, type(self))


		def __getitem__(self, slc):
			return OddDenseTensor(self.value[slc])
		### do not test equal tfe
		def __add__(self, other):
			return self.add(other)

		def __sub__(self, other):
			return self.sub(other)

		def add(self, other):
			"""Add other to this tensor."""
			x, y = _lift(self, other)
			bitlength = math.ceil(math.log2(master_factory.modulus))

			x_value = x.value
			y_value = y.value
			z = x_value + y_value
			wrapped_around = _lessthan_as_unsigned(-2 - y_value, x_value, bitlength)
			z += wrapped_around


			# the below avoids redundant seed expansion; can be removed once
			# we have a (per-device) caching mechanism in place

			### I don't understand
			# we want to compute whether we wrapped around, ie `pos(x) + pos(y) >= m - 1`,
			# for correction purposes which, since `m - 1 == 1` for signed integers, can be
			# rewritten as:
			#	 -> `pos(x) >= m - 1 - pos(y)`
			#	 -> `m - 1 - pos(y) - 1 < pos(x)`
			#	 -> `-1 - pos(y) - 1 < pos(x)`
			#	 -> `-2 - pos(y) < pos(x)`


			return OddDenseTensor(z)


		def sub(self, other):
			"""Subtract other from this tensor."""
			x, y = _lift(self, other)
			bitlength = math.ceil(math.log2(master_factory.modulus))

			x_value = x.value
			y_value = y.value
			z = x_value - y_value
			wrapped_around = _lessthan_as_unsigned(-2 - y_value, x_value, bitlength)
			z += wrapped_around

			return OddDenseTensor(z)

### ----------------- testing here ----------------------###
		### these fun not modify
		def bits(self, factory=None):
			if factory is None:
				return OddDenseTensor(binarize(self.value))
			return factory.tensor(binarize(self.value))

		def cast(self, factory):
			if factory is master_factory:
				return self
			return factory.tensor(self.value)

	class OddDenseTensor(OddTensor):
		"""
		Represents a tensor with explicit values, as opposed to OddUniformTensor
		with implicit values.

		Internal use only and assume that invalid values have already been mapped.
		"""
		## ok
		def __init__(self, value):
			assert isinstance(value, torch.Tensor)
			self._value = value

		@property
		def value(self) -> torch.Tensor:
			return self._value

		@property
		def shape(self):
			return self._value.shape

		def size(self, dim = None):
			if dim is None:
				return self._value.size()
			else:
				return self._value.size(dim)		

		# def size(self, dim = None):
		# 	if dim is None:
		# 		return self.size()
		# 	else:
		# 		return self.size(dim)

		@property
		def support(self):
			return [self._value]

	### not modify
	# class OddUniformTensor(OddTensor):
	# 	"""
	# 	Represents a tensor with uniform values defined implicitly through a seed.

	# 	Internal use only.
	# 	"""

	# 	def __init__(self, shape, seed):
	# 		self._seed = seed
	# 		self._shape = shape

	# 	@property
	# 	def shape(self):
	# 		return self._shape

	# 	@property
	# 	def value(self) -> tf.Tensor:
	# 		# TODO(Morten) result should be stored in a (per-device) cache
	# 		with tf.name_scope('expand-seed'):
	# 			sampler = partial(secure_random.seeded_random_uniform, seed=self._seed)
	# 			value = _construct_value_from_sampler(sampler=sampler, hape=self._shape)
	# 			return value
	# 	@property
	# 	def support(self):
	# 		return [self._seed]




	### this fun is not compete by author
	def _lift(x, y) -> Tuple[OddTensor, OddTensor]:
		"""
		Attempts to lift x and y to compatible OddTensors for further processing.
		"""

		if isinstance(x, OddTensor) and isinstance(y, OddTensor):
			assert x.factory == y.factory, "Incompatible types: {} and {}".format(
					x.factory, y.factory)
			return x, y


		### there is not compete by author
		if isinstance(x, OddTensor):
			if isinstance(y, int):
				print('ccc')
				return x, x.factory.tensor(np.array([y]))
		if isinstance(y, OddTensor):
			if isinstance(x, int):
				return y.factory.tensor(np.array([x])), y

		raise TypeError("Don't know how to lift {} {}".format(type(x), type(y)))

	def _construct_value_from_sampler(sampler, shape):
		"""Sample from sampler and correct for the modified dtype."""
		# to get uniform distribution over [min, max] without -1 we sample
		# [min+1, max] and shift negative values down by one

		if NATIVE_TYPE is torch.int32:
			minval = -(2 ** 31)
			maxval = (2 ** 31) - 1 
		elif NATIVE_TYPE is torch.int64:
			minval = -(2 ** 63)
			maxval = (2 ** 63) - 1 

		unshifted_value = torch.randint(low=minval + 1, high = maxval, size = shape, dtype=NATIVE_TYPE, layout=torch.strided, 
					device=None, requires_grad=False)

		value = torch.where(unshifted_value < 0, unshifted_value + 1, unshifted_value)
		return value


	def _lessthan_as_unsigned(x, y, bitlength):
		"""
		Performs comparison `x < y` on signed integers *as if* they were unsigned,
		e.g. `1 < -1`. Taken from Section 2-12, page 23, of
		[Hacker's Delight](https://www.hackersdelight.org/).
		"""
		x_np = x.numpy()
		y_np = y.numpy()

		not_x = np.invert(x_np)
		lhs = np.bitwise_and(not_x, y_np)
		rhs = np.bitwise_and(np.bitwise_or(not_x, y_np), x_np - y_np)
		z = np.right_shift(np.bitwise_or(lhs, rhs), bitlength - 1)

		z = np.bitwise_and(z, np.ones_like(z))

		return torch.from_numpy(z) 

	def _map_minusone_to_zero(value):
		""" Maps all -1 values to zero. """
		zeros = torch.zeros_like(value)
		return torch.where(value == -1, zeros, value)


	return master_factory


oddint32_factory = odd_factory(torch.int32)
oddint64_factory = odd_factory(torch.int64)


# print(type(x))
if __name__ == '__main__':

	x = oddint64_factory.tensor(torch.tensor([-1, 17], dtype=torch.int64))
	y = oddint64_factory.tensor(torch.tensor([-1, 2**63 -1], dtype=torch.int64))
	print(x + y)