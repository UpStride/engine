import os
import unittest
import tempfile
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_math_ops
from upstride import generic_layers
from upstride.generic_layers import _ga_multiply_get_index, upstride_type, unit_multiplier, reorder
from upstride.type2.tf.keras.utils import quaternion_mult1, quaternion_mult2, multiply_by_a1, multiply_by_a2, quaternion_mult_naive, quaternion_mult_cpp
from upstride.type2.tf.keras.layers import TF2Upstride as QTF2Upstride
from upstride.type2.tf.keras.layers import Upstride2TF as QUpstride2TF
from upstride.type2.tf.keras.layers import determine_norm_order
from upstride.type2.tf.keras import layers as type2_layers
from upstride.test_batchnorm import TestQuaternionBN, TestComplexBN
from upstride.type2.tf.keras.test_custom_ops import TestCustomOpPythonBackprop, TestCustomOpCpp, TestCustomOpCppBackprop
from upstride.type1.tf.keras.test_activations import TestActivationCos, TestActivationPow2
from upstride.test_initializers import TestCInitializer, TestHInitializer, TestIndependentFilter

class TestGAMultiplication(unittest.TestCase):
  def test_ga_multiply_get_index(self):
    s, index = _ga_multiply_get_index("123", "12")
    self.assertEqual(s, -1)
    self.assertEqual(index, "3")
    s, index = _ga_multiply_get_index("13", "12")
    self.assertEqual(s, 1)
    self.assertEqual(index, "23")
    s, index = _ga_multiply_get_index("3", "2")
    self.assertEqual(s, -1)
    self.assertEqual(index, "23")
    s, index = _ga_multiply_get_index("32", "32")
    self.assertEqual(s, -1)
    self.assertEqual(index, "")
    s, index = _ga_multiply_get_index("2", "2")
    self.assertEqual(s, 1)
    self.assertEqual(index, "")

  def test_unit_multiplier(self):
    generic_layers.change_upstride_type(3, ["", "1", "2", "3", "12", "13", "23", "123"], (3, 0, 0))
    # order : (scalar, e1, e2, e3, e12, e13, e23, e123)
    self.assertEqual(unit_multiplier(0, 0), (0, 1))  # 1*1 = 1
    self.assertEqual(unit_multiplier(3, 3), (0, 1))  # e_3*e_3 = 1
    self.assertEqual(unit_multiplier(4, 5), (6, -1))  # e12 * e13 = -e23
    self.assertEqual(unit_multiplier(6, 7), (1, -1))  # e23 * e123 = -e1

  def test_reorder(self):
    inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
    self.assertEqual(reorder(inputs), [[1, 5], [2, 6], [3, 7], [4, 8]])

  def test_bias_undefined(self):
    generic_layers.upstride_type = 1
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = generic_layers.TF2Upstride()(inputs)
    x = generic_layers.Conv2D(4, (3, 3))(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 3)  # input, conv, bias
    self.assertEqual(model.count_params(), 9*4*3+4)

  def test_bias_defined_false(self):
    generic_layers.upstride_type = 1
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = generic_layers.TF2Upstride()(inputs)
    x = generic_layers.Conv2D(4, (3, 3), use_bias=False)(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 2)  # input, conv
    self.assertEqual(model.count_params(), 9*4*3)

  def test_bias_defined_true(self):
    generic_layers.upstride_type = 1
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = generic_layers.TF2Upstride()(inputs)
    x = generic_layers.Conv2D(4, (3, 3), use_bias=True)(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 3)  # input, conv, bias
    self.assertEqual(model.count_params(), 9*4*3+4)


class TestQuaternionTF2Upstride(unittest.TestCase):
  def test_QTF2Upstride(self):
    inputs = tf.convert_to_tensor([[[[1, 3, 4]]]])
    o = QTF2Upstride()(inputs)
    self.assertEqual(type(o), list)
    self.assertEqual(o[0].shape, (1, 1, 1, 3))

    o = QTF2Upstride("joint")(inputs)
    self.assertEqual(type(o), list)
    self.assertEqual(o[0].shape, (1, 1, 1, 1))
    self.assertEqual(o[1].shape, (1, 1, 1, 1))
    self.assertEqual(o[2].shape, (1, 1, 1, 1))
    self.assertEqual(o[3].shape, (1, 1, 1, 1))

    # go to channel fist and continue the testing
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.convert_to_tensor([[[[1]], [[3]], [[4]]]])
    self.assertEqual(inputs.shape, (1, 3, 1, 1))

    o = QTF2Upstride()(inputs)
    self.assertEqual(type(o), list)
    self.assertEqual(o[0].shape, (1, 3, 1, 1))

    o = QTF2Upstride("joint")(inputs)
    self.assertEqual(type(o), list)
    self.assertEqual(o[0].shape, (1, 1, 1, 1))
    self.assertEqual(o[1].shape, (1, 1, 1, 1))
    self.assertEqual(o[2].shape, (1, 1, 1, 1))
    self.assertEqual(o[3].shape, (1, 1, 1, 1))

    o = QTF2Upstride("grayscale")(inputs)
    self.assertEqual(type(o), list)
    self.assertEqual(o[0].shape, (1, 1, 1, 1))
    self.assertEqual(o[1].shape, (1, 1, 1, 1))
    self.assertEqual(o[2].shape, (1, 1, 1, 1))
    self.assertEqual(o[3].shape, (1, 1, 1, 1))

    
    tf.keras.backend.set_image_data_format('channels_last')


class TestQuaternionMult(unittest.TestCase):
  def test_multiply_by_a1(self):
    self.assertEqual(multiply_by_a1([1, 0, 0, 0]), [1, 1, 1, 1])
    self.assertEqual(multiply_by_a1([0, 1, 0, 0]), [1, -1, 1, -1])

  def test_multiply_by_a2(self):
    self.assertEqual(multiply_by_a2([1, 0, 0, 0]), [1, 1, 1, 1])
    self.assertEqual(multiply_by_a2([0, 1, 0, 0]), [1, -1, 1, -1])

  def test_quaternion_mult1(self):
    def op(x, y): return x*y
    self.assertEqual(quaternion_mult1(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(quaternion_mult1(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(quaternion_mult1(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(quaternion_mult1(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(quaternion_mult1(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

  def test_quaternion_mult2(self):
    def op(x, y): return x*y
    self.assertEqual(quaternion_mult2(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(quaternion_mult2(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(quaternion_mult2(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(quaternion_mult2(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(quaternion_mult2(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

  def test_quaternion_mult_naive(self):
    def op(x, y): return x*y
    self.assertEqual(quaternion_mult_naive(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(quaternion_mult_naive(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(quaternion_mult_naive(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(quaternion_mult_naive(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(quaternion_mult_naive(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

  def test_quaternion_mult_cpp(self):
    def op(x, y): return x*y
    def convert_to_list(op_result): return [op_result[i] for i in range(len(op_result))]
    self.assertEqual(convert_to_list(quaternion_mult_cpp(op,  [1., 0., 0., 0.], [1., 0., 0., 0.])), [1., 0., 0., 0.])
    self.assertEqual(convert_to_list(quaternion_mult_cpp(op,  [1., 0., 0., 0.], [0., 2., 0., 0.])), [0., 2., 0., 0.])
    self.assertEqual(convert_to_list(quaternion_mult_cpp(op,  [0., 2., 2., 0.], [0., 2., 0., 0.])), [-4., 0., 0., -4.])
    self.assertEqual(convert_to_list(quaternion_mult_cpp(op,  [1., 2., 0., 3.], [0., 2., 2., 0.])), [-4., -4., 8., 4.])
    self.assertEqual(convert_to_list(quaternion_mult_cpp(op,  [1., 2., 3., 4.], [5., 6., 7., 8.])), [-60., 12., 30., 24.])


class TestConv2DQuaternion(unittest.TestCase):
  def test_conv2d(self):
    generic_layers.upstride_type = 2
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = type2_layers.TF2Upstride()(inputs)
    x = type2_layers.Conv2D(4, (3, 3), use_bias=True)(x)
    x = type2_layers.Upstride2TF("take_first")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 2)
    self.assertEqual(model.count_params(), (9*4*3+4)*4)

  def test_export(self):
    generic_layers.upstride_type = 2
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = type2_layers.TF2Upstride()(inputs)
    x = type2_layers.Conv2D(4, (3, 3), use_bias=True)(x)
    x = type2_layers.DepthwiseConv2D(4, (3, 3), use_bias=True)(x)
    x = type2_layers.Upstride2TF("take_first")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    dest = tempfile.mkdtemp()
    tf.saved_model.save(model, dest)
    listdir = os.listdir(dest)
    listdir.sort()
    self.assertEqual(listdir, ['assets', 'saved_model.pb', 'variables'])
    shutil.rmtree(dest)

class TestQuaternionUpstride2TF(unittest.TestCase):
  def  test_determine_norm_order(self):
    
    self.assertEqual(np.inf, determine_norm_order('norm_inf'))
    self.assertEqual(1, determine_norm_order('norm_1'))
    self.assertEqual(2, determine_norm_order('norm_2'))

    self.assertRaises(ValueError, determine_norm_order, 'norm_abc')
    self.assertRaises(ValueError, determine_norm_order, 'norm_-1')

  def  test_take_first(self):
    inputs = [tf.convert_to_tensor([[[[1, 3, 4]]]]), tf.convert_to_tensor([[[[1, 3, 4]]]])]

    output = QUpstride2TF("take_first")(inputs)

    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))
    # check the 1st tensor of inputs list and output tensor
    self.assertEqual(inputs[0].numpy().tolist(), output.numpy().tolist())

  def  test_concat(self):
    inputs = [tf.convert_to_tensor([[[[1, 3, 4]]]]), tf.convert_to_tensor([[[[1, 3, 4]]]])]

    output = QUpstride2TF("concat")(inputs)

    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))
    # check the concatenation of input tensors list and output tensor
    self.assertEqual(np.concatenate([inp.numpy() for inp in  inputs],  -1).tolist(), output.numpy().tolist())
     # check the size of concatenation dimension
    self.assertEqual(inputs[0].numpy().shape[-1]*len(inputs), output.numpy().shape[-1])

  def  test_max_pool(self):
    inputs = [tf.convert_to_tensor([[[[1, 3, 4]]]]), tf.convert_to_tensor([[[[2, 1, 6]]]])]

    output = QUpstride2TF("max_pool")(inputs)

    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))

    # check the elements after max pooling
    input_stack =  np.stack([inp.numpy() for inp in  inputs],  -1)
    self.assertEqual(np.max(input_stack, -1).tolist(), output.numpy().tolist())

  def  test_avg_pool(self):
    inputs = [tf.convert_to_tensor([[[[1., 3., 4.]]]]), tf.convert_to_tensor([[[[2., 1., 6.]]]])]

    output = QUpstride2TF("avg_pool")(inputs)

    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))

    # check the elements after average pooling
    input_stack =  np.stack([inp.numpy() for inp in  inputs],  -1)
    self.assertEqual(np.mean(input_stack, -1).tolist(), output.numpy().tolist())


  def test_p_norm(self):
    inputs = [tf.convert_to_tensor([[[[1., 3., 4.]]]]), tf.convert_to_tensor([[[[2., 1., 6.]]]])]

    output = QUpstride2TF("norm_1")(inputs)

    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))

    # check the elements after average pooling
    input_stack =  np.stack([inp.numpy() for inp in  inputs],  -1)
    self.assertEqual(np.linalg.norm(input_stack, ord=1, axis=-1).tolist(), output.numpy().tolist())

    # check norm-2
    self.assertEqual(np.linalg.norm(input_stack, ord=2, axis=-1).tolist(), QUpstride2TF("norm_2")(inputs).numpy().tolist())

    # check norm-inf
    self.assertEqual(np.linalg.norm(input_stack, ord=np.inf, axis=-1).tolist(), QUpstride2TF("norm_inf")(inputs).numpy().tolist())

  def test_attention(self):
    inputs = [tf.convert_to_tensor([[1., 2., 3.]]), tf.convert_to_tensor([[1., 2., 3.]])]

    ## Check for normal attention
    output = QUpstride2TF("attention")(inputs)
    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(output))
    # check values
    self.assertEqual(inputs[0].numpy().tolist(), output.numpy().tolist())

    ## Check for gated attention
    gated_output = QUpstride2TF("gated_attention")(inputs)
    # check the inputs list and out tensor type
    self.assertNotEqual(type(inputs), type(gated_output))
    # check the 1st tensor of inputs list and out tensor type
    self.assertEqual(type(inputs[0]), type(gated_output))
    # check values
    self.assertEqual(inputs[0].numpy().tolist(), gated_output.numpy().tolist())

    # check rank of input tensors with rank 1
    inputs1 = [tf.convert_to_tensor([1., 2., 3.]), tf.convert_to_tensor([1., 2., 3.])]
    self.assertRaises(TypeError, QUpstride2TF("attention"), inputs1)

    # check rank of input tensors with rank 5
    inputs2 = [tf.convert_to_tensor([[[[[1., 2., 3.]]]]]), tf.convert_to_tensor([[[[[1., 2., 3.]]]]])]
    self.assertRaises(TypeError, QUpstride2TF("attention"), inputs2)

if __name__ == "__main__":
  unittest.main()
