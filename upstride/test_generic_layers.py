import unittest
import tensorflow as tf
from upstride.generic_layers import _ga_multiply_get_index, unit_multiplier, change_upstride_type
from upstride import generic_layers

class TestGAMultiplication(unittest.TestCase):
  def test_ga_multiply_get_index(self):
    change_upstride_type(3, ["", "1", "2", "3", "12", "13", "23", "123"], (3, 0, 0))
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

  def test_bias_undefined(self):
    tf.keras.backend.set_image_data_format('channels_first')
    change_upstride_type(1, ["", "12"], (2, 0, 0))
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = generic_layers.TF2Upstride('basic')(inputs)
    x = generic_layers.Conv2D(4, (3, 3))(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, tf2upstride, conv, split
    # one kernel has shape 3*3*3*4=108. we have one real and one imaginary, so 216 parameters.
    # one bias for real and one for imaginary, 224 parameters
    self.assertEqual(model.count_params(), 224)

  def test_bias_defined_false(self):
    tf.keras.backend.set_image_data_format('channels_first')
    change_upstride_type(1, ["", "12"], (2, 0, 0))
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = generic_layers.TF2Upstride('basic')(inputs)
    x = generic_layers.Conv2D(4, (3, 3), use_bias=False)(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, tf2upstride, conv, split
    self.assertEqual(model.count_params(), 216)

  def test_bias_defined_true(self):
    tf.keras.backend.set_image_data_format('channels_first')
    change_upstride_type(1, ["", "12"], (2, 0, 0))
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = generic_layers.TF2Upstride('basic')(inputs)
    x = generic_layers.Conv2D(4, (3, 3), use_bias=True)(x)
    x = generic_layers.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4) # input, tf2upstride, conv, split
    self.assertEqual(model.count_params(), 224)
