import unittest
import random
import tensorflow as tf
import numpy as np
from math import pi
from .activations import cos_fn, cos_fn_grad, pow2_fn, pow2_fn_grad

def get_tf_grad(z, fn, *args):
  a, b = tf.constant(z[0]), tf.constant(z[1])
  with tf.GradientTape(persistent=True) as g:
    g.watch(a)
    g.watch(b)
    y = fn([a,b],*args)
    dy_da, dy_db = g.gradient(y, a), g.gradient(y, b)

  return [dy_da, dy_db]
  

class TestActivationCos(unittest.TestCase):

  def test_forward_scalar(self):
    a, b = 0.0, 0.0
    target_a, target_b = 1, 0
    out = cos_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_forward_zeros(self):
    a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
    target_a, target_b = np.ones([2,3,4]), np.zeros([2,3,4])
    out = cos_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_forward_90deg(self):
    a, b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
    target_a, target_b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
    out = cos_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_backward_zeros(self):
    a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
    target_a, target_b = np.ones([2,3,4]), np.ones([2,3,4])

    ## Manual gradients
    target_da, target_db = cos_fn_grad([a,b])

    ## TF computed gradient
    dy_da, dy_db = get_tf_grad([a,b], cos_fn)

    self.assertTrue(np.array_equal(dy_da,target_da) and np.array_equal(dy_db ,target_db) \
    and np.array_equal(dy_da,target_a) and np.array_equal(dy_db,target_b))

  def test_backward_90deg(self):
    a, b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
    target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])

    ## Manual gradients
    target_da, target_db = cos_fn_grad([a,b])

    ## TF computed gradient
    dy_da, dy_db = get_tf_grad([a,b], cos_fn)

    self.assertTrue(np.array_equal(dy_da,target_da) and np.array_equal(dy_db ,target_db) \
    and np.array_equal(dy_da,target_a) and np.array_equal(dy_db ,target_b))


class TestActivationPow2(unittest.TestCase):

  def test_forward_scalar(self):
    a, b = 0.0, 0.0
    target_a, target_b = 0.0, 0.0
    out = pow2_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_forward_zeros(self):
    a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
    target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])
    out = pow2_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_forward_ones(self):
    a, b = np.ones([2,3,4]), np.ones([2,3,4])
    target_a, target_b = np.zeros([2,3,4]), 2*np.ones([2,3,4])
    out = pow2_fn([a,b])
    self.assertTrue(np.array_equal(out[0].numpy(),target_a) and np.array_equal(out[1].numpy(),target_b))

  def test_backward_zeros(self):
    a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
    target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])
    
    ## Manual gradients
    target_da, target_db = pow2_fn_grad([a,b])

    ## TF computed gradient
    dy_da, dy_db = get_tf_grad([a,b], pow2_fn)

    self.assertTrue(np.array_equal(dy_da, target_da) and np.array_equal(dy_db ,target_db))
    self.assertTrue(np.array_equal(dy_da, target_a) and np.array_equal(dy_db ,target_b))

  def test_backward_ones(self):
    a, b = np.ones([2,3,4]), np.ones([2,3,4])
    target_a, target_b = 4*np.ones([2,3,4]), np.zeros([2,3,4])
    ## Manual gradients
    target_da, target_db = pow2_fn_grad([a,b])

    ## TF computed gradient
    dy_da, dy_db = get_tf_grad([a,b], pow2_fn)

    self.assertTrue(np.array_equal(dy_da, target_da) and np.array_equal(dy_db ,target_db))
    self.assertTrue(np.array_equal(dy_da, target_a) and np.array_equal(dy_db ,target_b))

  def test_backward_alphas(self):
    a, b = np.ones([2,3,4]), np.ones([2,3,4])

    for _ in range(10):

      alpha = random.uniform(0.5,5.0)

      ## Manual gradients
      target_da, target_db = pow2_fn_grad([a,b], alpha)

      ## TF computed gradient
      dy_da, dy_db = get_tf_grad([a,b], pow2_fn, alpha)

      self.assertTrue(np.array_equal(dy_da, target_da) and np.array_equal(dy_db ,target_db))

if __name__ == '__main__':
    unittest.main()
