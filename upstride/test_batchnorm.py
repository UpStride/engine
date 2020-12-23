import unittest
import tensorflow as tf
import numpy as np
from .batchnorm import BatchNormalizationH, BatchNormalizationC


class TestQuaternionBN(unittest.TestCase):
  def test_init(self):
    """ Basic test to see if we can call BN on a simple case
    """
    inputs = tf.convert_to_tensor([[[[1., 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                    [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]]])
    self.assertEqual(inputs.shape, (4, 2, 3, 5))
    bn_layer = BatchNormalizationH()
    outputs = bn_layer(inputs, training=True)
    self.assertEqual(outputs.shape, (4, 2, 3, 5))
    self.assertTrue(np.array_equal(outputs, np.zeros((4, 2, 3, 5))))

  def test_compute_sqrt_inv(self):
    """ 
    From a matrix M,
    Call compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    inputs = tf.random.normal(shape=(4, 2, 3, 5))
    bn_layer = BatchNormalizationH()
    # call it once to init everything
    bn_layer(inputs, training=False)
    inputs = tf.split(inputs, 4, axis=0)
    _, _, v = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(1):
      v_matrix = np.array([[v['rr'][i], v['ri'][i], v['rj'][i], v['rk'][i]],
                           [v['ri'][i], v['ii'][i], v['ij'][i], v['ik'][i]],
                           [v['rj'][i], v['ij'][i], v['jj'][i], v['jk'][i]],
                           [v['rk'][i], v['ik'][i], v['jk'][i], v['kk'][i]]])

      w_matrix = np.array([[w['rr'][i],          0, 0, 0],
                           [w['ri'][i], w['ii'][i], 0, 0],
                           [w['rj'][i], w['ij'][i], w['jj'][i], 0],
                           [w['rk'][i], w['ik'][i], w['jk'][i], w['kk'][i]]])
      should_be_id = np.dot(np.dot(w_matrix.T, w_matrix), v_matrix)
      for i in range(4):
        for j in range(4):
          if i == j:
            self.assertAlmostEqual(should_be_id[i][j], 1, 5)
          else:
            self.assertAlmostEqual(should_be_id[i][j], 0, 5)


class TestComplexBN(unittest.TestCase):
  def test_init(self):
    inputs = tf.convert_to_tensor([[[[1., 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                    [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]]])
    self.assertEqual(inputs.shape, (2, 2, 3, 5))
    bn_layer = BatchNormalizationC()
    outputs = bn_layer(inputs, training=True)
    self.assertEqual(outputs.shape, (2, 2, 3, 5))
    self.assertTrue(np.array_equal(outputs, np.zeros((2, 2, 3, 5))))

  def test_compute_sqrt_inv(self):
    """ 
    From a matrix M,
    Call a compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    inputs = tf.random.normal(shape=(10, 5, 3, 2))
    bn_layer = BatchNormalizationC()
    # call it once to init everything
    bn_layer(inputs, training=False)
    inputs = tf.split(inputs, 2, axis=0)
    _, _, v = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(5):
      v_matrix = np.array([[v['rr'][i], v['ri'][i]], [v['ri'][i], v['ii'][i]]])
      w_matrix = np.array([[w['rr'][i], w['ri'][i]], [w['ri'][i], w['ii'][i]]])
      should_be_id = np.dot(np.dot(w_matrix, w_matrix), v_matrix)
      self.assertAlmostEqual(should_be_id[0][0], 1, 6)
      self.assertAlmostEqual(should_be_id[1][1], 1, 6)
      self.assertAlmostEqual(should_be_id[1][0], 0, 6)
      self.assertAlmostEqual(should_be_id[0][1], 0, 6)
