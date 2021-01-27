import unittest
import random
import tensorflow as tf
import numpy as np
from math import pi
from .activations import relu_linebound_fn, swish_linebound_fn
from tensorflow.math import sin, cos, atan2, pow, sqrt, sigmoid, greater_equal, multiply

def get_tf_grad(z, fn, *args):
  a, b = tf.constant(z[0]), tf.constant(z[1])
  with tf.GradientTape(persistent=True) as g:
    g.watch(a)
    g.watch(b)
    y = fn([a,b],*args)
    dy_da, dy_db = g.gradient(y, a), g.gradient(y, b)

  return [dy_da, dy_db]

def NHWC_relu_linebound_fn(z, l):
  """
  Former version of the linebound function implemented for the old engine => channel last format.
  Use this function to make sure we get the same results with the channel first implementation.
  """

  z = tf.transpose(z, [0, 2, 3, 1])
  a, b = tf.split(z, 2, axis=0)

  ## Original function
  # START
  #a, b = z[0], z[1]
  a = tf.transpose(a, [0, 3, 1, 2])
  b = tf.transpose(b, [0, 3, 1, 2])

  z_new = tf.concat([tf.expand_dims(a,axis=0), tf.expand_dims(b,axis=0)], axis=0)
  lam = tf.tensordot(l, z_new, axes=1)
  condition = greater_equal(lam, 0)

  real = tf.where(condition, tf.tanh(lam)*a, 0)
  imag = tf.where(condition, tf.tanh(lam)*b, 0)
  real = tf.transpose(real, [0, 2, 3, 1])
  imag = tf.transpose(imag, [0, 2, 3, 1])
  # END

  return tf.transpose(tf.concat([real, imag], axis=0), [0, 3, 1, 2])

def NHWC_swish_linebound_fn(z, l):
  """
  Former version of the linebound function implemented for the old engine => channel last format.
  Use this function to make sure we get the same results with the channel first implementation.
  """

  z = tf.transpose(z, [0, 2, 3, 1])
  a, b = tf.split(z, 2, axis=0)

  ## Original function
  # START
  #a, b = z[0], z[1]
  a = tf.transpose(a, [0, 3, 1, 2])
  b = tf.transpose(b, [0, 3, 1, 2])
  phase = atan2(b,a)
  norm = sqrt(pow(a,2)+pow(b,2))

  z_new = tf.concat([tf.expand_dims(a,axis=0), tf.expand_dims(b,axis=0)], axis=0)
  lam = tf.tensordot(l, z_new, axes=1)
  condition = greater_equal(lam, 0)

  smooth_norm = tf.nn.sigmoid(lam)*norm
  smooth_phase = tf.nn.sigmoid(lam)*phase

  real = multiply(smooth_norm,cos(smooth_phase))
  imag = multiply(smooth_norm,sin(smooth_phase))
  real = tf.transpose(real, [0, 2, 3, 1])
  imag = tf.transpose(imag, [0, 2, 3, 1])

  # END

  return tf.transpose(tf.concat([real, imag], axis=0), [0, 3, 1, 2])


class TestLineBound(unittest.TestCase):

  def test_relu_forward_negative(self):
    z = tf.constant((-1000)*np.ones([4,4,4,4], dtype=np.float32))
    l = tf.constant([1.,1.])
    target = np.zeros([4,4,4,4], dtype=np.float32)
    out = relu_linebound_fn(z,l)
    self.assertTrue(np.allclose(out.numpy(),target, rtol=1e-03, atol=1e-03))

  def test_relu_forward_positive(self):
    z = tf.constant((+1000)*np.ones([4,4,4,4], dtype=np.float32))
    l = tf.constant([1.,1.])
    target = (+1000)*np.ones([4,4,4,4], dtype=np.float32)
    out = relu_linebound_fn(z,l)
    self.assertTrue(np.allclose(out.numpy(),target, rtol=1e-03, atol=1e-03))

  def test_relu_old_version(self):
    l = tf.constant([1.,1.])
    for i in range(10):
      z = tf.random.uniform([4,4,4,4], seed=i)
      target = NHWC_relu_linebound_fn(z,l)
      out = relu_linebound_fn(z,l)
      self.assertTrue(np.allclose(out.numpy(),target))

  def test_swish_forward_negative(self):
    z = tf.constant((-1000)*np.ones([4,4,4,4], dtype=np.float32))
    l = tf.constant([1.,1.])
    target = np.zeros([4,4,4,4], dtype=np.float32)
    out = swish_linebound_fn(z,l)
    self.assertTrue(np.allclose(out.numpy(),target, rtol=1e-03, atol=1e-03))

  def test_swish_forward_positive(self):
    z = tf.constant((+1000)*np.ones([4,4,4,4], dtype=np.float32))
    l = tf.constant([1.,1.])
    target = (+1000)*np.ones([4,4,4,4], dtype=np.float32)
    out = swish_linebound_fn(z,l)
    self.assertTrue(np.allclose(out.numpy(),target, rtol=1e-03, atol=1e-03))

  def test_swish_old_version(self):
    l = tf.constant([1.,1.])
    for i in range(10):
      z = tf.random.uniform([4,4,4,4], seed=i)
      target = NHWC_swish_linebound_fn(z,l)
      out = swish_linebound_fn(z,l)
      self.assertTrue(np.allclose(out.numpy(),target))