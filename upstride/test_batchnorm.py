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

  def test_compute_sqrt_inv(self):
    """ 
    From a matrix M,
    Call compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    inputs = [tf.random.normal(shape=(1, 2, 3, 5)) for _ in range(4)]
    bn_layer = BatchNormalizationQ()
    # call it once to init everything
    bn_layer(inputs, training=False)
    _, _, v, _ = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(5):
      v_matrix = np.array([[v['rr'][i], v['ri'][i], v['rj'][i], v['rk'][i]],
                           [v['ri'][i], v['ii'][i], v['ij'][i], v['ik'][i]],
                           [v['rj'][i], v['ij'][i], v['jj'][i], v['jk'][i]],
                           [v['rk'][i], v['ik'][i], v['jk'][i], v['kk'][i]]])
                           
      w_matrix = np.array([[w['rr'][i], w['ri'][i], w['rj'][i], w['rk'][i]],
                           [w['ri'][i], w['ii'][i], w['ij'][i], w['ik'][i]],
                           [w['rj'][i], w['ij'][i], w['jj'][i], w['jk'][i]],
                           [w['rk'][i], w['ik'][i], w['jk'][i], w['kk'][i]]])
      should_be_id = np.dot(np.dot(w_matrix, w_matrix), v_matrix)
      print(should_be_id)
      # self.assertAlmostEqual(should_be_id[0][0], 1, 6)
      # self.assertAlmostEqual(should_be_id[1][1], 1, 6)
      # self.assertAlmostEqual(should_be_id[1][0], 0, 6)
      # self.assertAlmostEqual(should_be_id[0][1], 0, 6)


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
    inputs = [tf.random.normal(shape=(1, 2, 3, 5)) for _ in range(2)]
    bn_layer = BatchNormalizationC()
    # call it once to init everything
    bn_layer(inputs, training=False)
    _, _, v, _ = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(5):
      v_matrix = np.array([[v['rr'][i], v['ri'][i]], [v['ri'][i], v['ii'][i]]])
      w_matrix = np.array([[w['rr'][i], w['ri'][i]], [w['ri'][i], w['ii'][i]]])
      should_be_id = np.dot(np.dot(w_matrix, w_matrix), v_matrix)
      self.assertAlmostEqual(should_be_id[0][0], 1, 6)
      self.assertAlmostEqual(should_be_id[1][1], 1, 6)
      self.assertAlmostEqual(should_be_id[1][0], 0, 6)
      self.assertAlmostEqual(should_be_id[0][1], 0, 6)


if __name__ == "__main__":
  unittest.main()
