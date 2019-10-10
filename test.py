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
# print(int32factory.modulus)

from torch_e.protocol.pond import pond
from torch_e.protocol.pond.pond import Pond
prot = Pond()
prot.server_0.device_name = 'cpu'
prot.server_1.device_name = 'cpu'
prot.triple_source.device_name = 'cpu'



# TestPond

print(prot)
expected = np.array([1234567.9875])
x = prot.define_constant(expected)
print(x.constant_on_0.value)
