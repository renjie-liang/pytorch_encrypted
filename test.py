import torch
import numpy as np
## test config

import torch
x = torch.tensor([1], dtype = torch.int64)
x = x.reshape(1,1)


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
from torch_e.protocol.pond.pond import Pond
prot = Pond()
prot.server_0.device_name = 'cpu'
prot.server_1.device_name = 'cpu'
prot.triple_source.device_name = 'cpu'


# TestPond

# print(prot)
expected = np.array([1,565564])
public_x = prot.define_from_numpy(expected)
# print(x.constant_on_0.value)
# print(x.constant_on_0.shape)


private_x = prot.define_private_variable(expected)

# print(private_x.share0.value)
# print(private_x.share1.value)

# public_x = private_x.reveal()

# print(public_x.value_on_0.value)

# TestPondPublicEqual

x = np.array([100, 200, 100, 300])
y = np.array([100, 100, 100, 200])

public_x = prot.numpy_to_public(x)
public_y = prot.numpy_to_public(y)


public_equal = prot.equal(public_x, public_y)
print(public_equal.value_on_0.value)

