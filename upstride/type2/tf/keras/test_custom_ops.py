import unittest
import tensorflow as tf
from .custom_ops import *


def get_tf_gradient(input_tf, function_tf):
  with tf.GradientTape(persistent=True) as gt:
    for e in input_tf:
      gt.watch(e)
      outputs = function_tf(*input_tf)
    dinput_tf = [gt.gradient(outputs, e) for e in input_tf]
  return dinput_tf


def get_py_gradient(input_py, function_py, backprop_function_py):
  outputs = function_py(*input_py)
  outputs_grad = [1] * len(outputs)
  dinput_py = backprop_function_py(*outputs_grad)
  return dinput_py


class TestCustomOpPythonBackprop(unittest.TestCase):
  """check that the backprop gives the same result as tensorflow
  """
  def test_upstride_inputs_backprop(self):
    a_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]
    a_py = [e.numpy()[0] for e in a_tf]

    dinput_tf = get_tf_gradient(a_tf, upstride_inputs_py)
    dinput_py = get_py_gradient(a_py, upstride_inputs_py, upstride_inputs_backprop_py)
    for i in range(len(dinput_tf)):
      self.assertEqual((dinput_py[i] - dinput_tf[i]).numpy()[0], 0.)

  def test_upstride_kernels_backprop(self):
    b_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]
    b_py = [e.numpy()[0] for e in b_tf]

    db_tf = get_tf_gradient(b_tf, upstride_kernels_py)
    db_py = get_py_gradient(b_py, upstride_kernels_py, upstride_kernels_backprop_py)
    for i in range(len(db_py)):
      self.assertEqual((db_py[i] - db_tf[i]).numpy()[0], 0.)

  def test_upstride_outputs_backprop(self):
    cp_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(8)]
    cp_py = [e.numpy()[0] for e in cp_tf]

    dcp_tf = get_tf_gradient(cp_tf, upstride_outputs_py)
    dcp_py = get_py_gradient(cp_py, upstride_outputs_py, upstride_outputs_backprop_py)
    for i in range(len(dcp_tf)):
      self.assertEqual((dcp_tf[i] - dcp_py[i]).numpy()[0], 0.)

class TestCustomOpCpp(unittest.TestCase):
  def test_upstride_inputs(self):
    a_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]
    output_cpp = upstride_inputs(*a_tf)
    output_py = upstride_inputs_py(*a_tf)
    for i in range(len(output_cpp)):
      self.assertEqual(output_cpp[i], output_py[i])

  def test_upstride_outputs(self):
    cp_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(8)]
    output_cpp = upstride_outputs(*cp_tf)
    output_py = upstride_outputs_py(*cp_tf)
    for i in range(len(output_cpp)):
      self.assertEqual(output_cpp[i], output_py[i])

  def test_upstride_kernels(self):
    b_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]
    output_cpp = upstride_kernels(*b_tf)
    output_py = upstride_kernels_py(*b_tf)
    for i in range(len(output_cpp)):
      self.assertEqual(output_cpp[i], output_py[i])

class TestCustomOpCppBackprop(unittest.TestCase):
  def test_upstride_inputs_backprop(self):
    a_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]

    dinput_tf = get_tf_gradient(a_tf, upstride_inputs_py)
    dinput_tf_cpp = get_tf_gradient(a_tf, upstride_inputs)
    for i in range(len(dinput_tf)):
      self.assertEqual((dinput_tf_cpp[i] - dinput_tf[i]).numpy()[0], 0.)

  def test_upstride_kernels_backprop(self):
    a_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(4)]

    dkernel_tf = get_tf_gradient(a_tf, upstride_kernels_py)
    dkernel_tf_cpp = get_tf_gradient(a_tf, upstride_kernels)
    for i in range(len(dkernel_tf)):
      self.assertEqual((dkernel_tf_cpp[i] - dkernel_tf[i]).numpy()[0], 0.)

  def test_upstride_outputs_backprop(self):
    a_tf = [tf.random.uniform((1,), dtype=tf.float32) for _ in range(8)]

    doutput_tf = get_tf_gradient(a_tf, upstride_outputs_py)
    doutput_tf_cpp = get_tf_gradient(a_tf, upstride_outputs)
    for i in range(len(doutput_tf)):
      self.assertEqual((doutput_tf_cpp[i] - doutput_tf[i]).numpy()[0], 0.)