# pylint: disable=missing-docstring
import unittest
# test module

import numpy as np
import torch

from odd_tensor import oddint64_factory


class TestOddImplicitTensor(unittest.TestCase):


  def test_add(self) -> None:

    # regular, overflow, underflow
    x = oddint64_factory.tensor(torch.tensor([2, -2], dtype=torch.int64))
    y = oddint64_factory.tensor(torch.tensor([3, 3], dtype=torch.int64))

    z = x + y

    expected = torch.tensor([5, 2], dtype = torch.int64)
    actual = z.value
    expected = expected.numpy()
    actual = actual.numpy()

    np.testing.assert_array_almost_equal(actual, expected, decimal=3)

  def test_sub(self) -> None:

    # regular, overflow, underflow
    x = oddint64_factory.tensor(torch.tensor([2, -2], dtype=torch.int64))
    y = oddint64_factory.tensor(torch.tensor([3, 3], dtype=torch.int64))

    z = x - y
    expected = torch.tensor([-1, -5], dtype = torch.int64)
    actual = z.value
    expected = expected.numpy()
    actual = actual.numpy()

    np.testing.assert_array_almost_equal(actual, expected, decimal=3)


if __name__ == '__main__':
  unittest.main()
