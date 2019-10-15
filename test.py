import torch
import numpy as np
## test config

# import torch
# x = torch.tensor([1], dtype = torch.int64)
# x = x.reshape(1,1)


# from torch_e.config import get_config
# from torch_e import config
# c = config.LocalConfig()
# c.add_player('rj')
# c.add_player('stt')


# p = c.get_players('rj,stt')

# print(p[0].name)
# print(p[0].index)
# print(p[0].device_name)
# print(p[0].host)

# p1 = c.get_player('rj')

# a = config.get_config()
# b = a.get_player('rj')
# print(b.device_name)



# test tensor
# from torch_e.tensor.fixed import FixedpointConfig, _validate_fixedpoint_config
# from torch_e.tensor.factory import AbstractTensor

# from torch_e.tensor import native
# int32factory = native.native_factory(torch.int32)
# sample = int32factory.sample_uniform([2,2])
# print(int32factory.modulus)
# print(sample.value)
from torch_e.protocol.pond import pond
from torch_e.protocol.protocol import get_protocol, set_protocol
from torch_e.protocol.pond.pond import Pond
prot = Pond()
prot.server_0.device_name = 'cpu'
prot.server_1.device_name = 'cpu'
prot.triple_source.device_name = 'cpu'
set_protocol(prot)


# TestPond

# print(prot)
# expected = np.array([1,565564])
# public_x = prot.define_from_numpy(expected)
# print(x.constant_on_0.value)
# print(x.constant_on_0.shape)


# private_x = prot.define_private_variable(expected)

# print(private_x.share0.value)
# print(private_x.share1.value)

# public_x = private_x.reveal()

# print(public_x.value_on_0.value)

# TestPondPublicEqual

# x = np.array([100, 200, 100, 300])
# y = np.array([100, 100, 100, 200])

# public_x = prot.numpy_to_public(x)
# public_y = prot.numpy_to_public(y)

# public_equal = prot.equal(public_x, public_y)
# print(public_equal.value_on_0.value)

# test three type public mul
# type 1 public * int
# x = np.array([100, 200, 100, 300])
# y = 2

# public_x = prot.numpy_to_public(x)
# z = public_x * y
# print(z)
# type 2 public * float
# type 3 public * public

# test private matmul
# private.mm(private)
# x = np.array([[1, 2],[3, 4]])
# y = np.array([[3, 2],[1, 4]])
# private_x = prot.private_tensor(x)
# private_y = prot.private_tensor(y)

# private_z = private_x.mm(private_y)
# public_z = private_z.reveal()
# print(public_z.value_on_0.value)
# print(public_z.value_on_1.value)

### test private mul public
### bug: precise problem
# x = np.array([[100, 200],[300, 400]])
# y = 0.25
# private_x = prot.private_tensor(x)


# private_z = private_x * y
# public_z = private_z.reveal()
# print(public_z.value_on_0.value)
# print(public_z.value_on_1.value)


# test private add
#private + private
# x = np.array([[1, 2],[3, 4]])
# y = np.array([[3, 2],[1, 4]])
# private_x = prot.private_tensor(x)
# private_y = prot.private_tensor(y)

# private_z = private_x + private_y
# public_z = private_z.reveal()
# print(public_z.value_on_0.value)
# print(public_z.value_on_1.value)



# # test_public_division
# x = np.array([100, 200, 300, 400])
# y = np.array([10, 20, 30, 20])

# private_x = prot.private_tensor(x)
# public_y = prot.public_from_numpy(y)
# z = private_x / public_y


### test fifo
# from torch_e.queue import fifo
# fifo.FIFOQueue()

### test layer
### test core
# from torch_e.layers import core

# test Dense

# from torch_e.layers.dense import Dense
# batch_size = 2
# in_channels = 3
# input_shape = [batch_size, in_channels]
# D = Dense(input_shape = input_shape, out_features = 4, transpose_input=False, transpose_weight=False)

# x = np.array([[1, 2, 3],[3,5,6]])
# private_x = prot.private_tensor(x)
# D.initialize()

# D.forward(private_x)

# test sigmoid
# from torch_e.layers.activation import Sigmoid
# batch_size = 2
# in_channels = 3
# input_shape = [batch_size, in_channels]
# D = Sigmoid(input_shape)

# x = np.array([[1, 2, 3],[3,5,6]])
# private_x = prot.private_tensor(x)
# D.forward(private_x)


# test Relu
# from torch_e.layers.activation import Relu
# batch_size = 2
# in_channels = 3
# input_shape = [batch_size, in_channels]
# D = Relu(input_shape)

# x = np.array([[1, 2, 3],[3,5,6]])
# private_x = prot.private_tensor(x)
# D.forward(private_x)

### test Tanh
# from torch_e.layers.activation import Tanh
# batch_size = 2
# in_channels = 3
# input_shape = [batch_size, in_channels]
# D = Tanh(input_shape)

# x = np.array([[1, 2, 3],[3,5,6]])
# private_x = prot.private_tensor(x)
# D.forward(private_x)

## test AveragePooling2D 
### bug: precise problem
# from torch_e.layers.pooling import AveragePooling2D
# batch_size = 1
# in_channels = 1
# h_in = 4
# w_in = 4
# input_shape = [batch_size, in_channels, h_in, w_in]
# x = np.random.randint(low = 0, high = 100, size = input_shape, dtype = np.int64)

# D = AveragePooling2D(input_shape = input_shape,
# 				pool_size = 2, 
# 				strides = None,
# 				padding ="SAME", channels_first = True)

# # print(D.get_output_shape())
# private_x = prot.private_tensor(x)
# pool_private_x = D.forward(private_x)

# print(x)
# temp = pool_private_x.reveal()
# print(temp.value_on_0.value)
# input()


### test MaxPooling2D 
### debug  Only SecureNN supports Max Pooling
# from torch_e.layers.pooling import MaxPooling2D
# batch_size, in_channels, h_in, w_in = 1, 1, 4, 4
# input_shape = [batch_size, in_channels, h_in, w_in]
# x = np.random.randint(low = 0, high = 100, size = input_shape, dtype = np.int64)

# D = MaxPooling2D(input_shape = input_shape,
# 				pool_size = 2, 
# 				strides = None,
# 				padding ="SAME", channels_first = True)

# # print(D.get_output_shape())
# private_x = prot.private_tensor(x)
# pool_private_x = D.forward(private_x)