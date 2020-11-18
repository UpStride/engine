import unittest
import tensorflow as tf
import numpy as np
from .batchnorm import BatchNormalizationQ, BatchNormalizationC


class TestQuaternionBN(unittest.TestCase):
  def test_init(self):
    """ Basic test to see if we can call BN on a simple case
    """
    inputs = tf.convert_to_tensor([[[[1., 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]],
                                    [[1, 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]]]])
    self.assertEqual(inputs.shape, (1, 2, 3, 5))
    inputs = [inputs for _ in range(4)]
    bn_layer = BatchNormalizationQ()
    outputs = bn_layer(inputs, training=False)
    self.assertEqual(len(outputs), 4)
    self.assertTrue(np.array_equal(outputs[0], np.zeros((1, 2, 3, 5))))

  def test_sqrt_inv(self):
    """ 
    From a matrix M,
    Call compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    pass

  def test_not_scaled(self):
    pass

  def test_not_center(self):
    pass

  def test_not_center_nor_scaled(self):
    pass


class TestComplexBN(unittest.TestCase):
  def test_init(self):
    inputs = tf.convert_to_tensor([[[[1., 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]],
                                    [[1, 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]]]])
    self.assertEqual(inputs.shape, (1, 2, 3, 5))
    inputs = [inputs for _ in range(2)]
    bn_layer = BatchNormalizationC()
    outputs = bn_layer(inputs, training=False)
    self.assertEqual(len(outputs), 2)
    self.assertTrue(np.array_equal(outputs[0], np.zeros((1, 2, 3, 5))))

  def test_compute_sqrt_inv(self):
    """ 
    From a matrix M,
    Call a compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    pass

  def test_not_scaled(self):
    pass

  def test_not_center(self):
    pass

  def test_not_center_nor_scaled(self):
    pass


if __name__ == "__main__":
  unittest.main()
