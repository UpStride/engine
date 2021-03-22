import unittest
import tensorflow as tf
from upstride.generic_layers import GenericLinear
import numpy as np
from upstride import generic_layers
from upstride.type1.tf.keras import layers as layers_t1


class UselessLayerType3(GenericLinear):
  def __init__(self):
    args = [10]
    super().__init__(tf.keras.layers.Dense, 3, ["", "1", "2", "3", "12", "13", "23", "123"], (3, 0, 0), *args)


class UselessLayerType1(GenericLinear):
  def __init__(self):
    args = [10]
    super().__init__(tf.keras.layers.Dense, 2, ["", "12"], (2, 0, 0), *args)


class TestGAMultiplication(unittest.TestCase):
  def test_ga_multiply_get_index(self):
    layer = UselessLayerType3()
    s, index = layer._ga_multiply_get_index("123", "12")
    self.assertEqual(s, -1)
    self.assertEqual(index, "3")
    s, index = layer._ga_multiply_get_index("13", "12")
    self.assertEqual(s, 1)
    self.assertEqual(index, "23")
    s, index = layer._ga_multiply_get_index("3", "2")
    self.assertEqual(s, -1)
    self.assertEqual(index, "23")
    s, index = layer._ga_multiply_get_index("32", "32")
    self.assertEqual(s, -1)
    self.assertEqual(index, "")
    s, index = layer._ga_multiply_get_index("2", "2")
    self.assertEqual(s, 1)
    self.assertEqual(index, "")

  def test_unit_multiplier(self):
    layer = UselessLayerType3()
    # order : (scalar, e1, e2, e3, e12, e13, e23, e123)
    self.assertEqual(layer.unit_multiplier(0, 0), (0, 1))  # 1*1 = 1
    self.assertEqual(layer.unit_multiplier(3, 3), (0, 1))  # e_3*e_3 = 1
    self.assertEqual(layer.unit_multiplier(4, 5), (6, -1))  # e12 * e13 = -e23
    self.assertEqual(layer.unit_multiplier(6, 7), (1, -1))  # e23 * e123 = -e1

  def test_bias_undefined(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = layers_t1.TF2Upstride('basic')(inputs)
    x = layers_t1.Conv2D(4, (3, 3))(x)
    x = layers_t1.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, tf2upstride, conv, split
    # one kernel has shape 3*3*3*4=108. we have one real and one imaginary, so 216 parameters.
    # one bias for real and one for imaginary, 224 parameters
    self.assertEqual(model.count_params(), 224)

  def test_bias_defined_channels_last(self):
    tf.keras.backend.set_image_data_format('channels_last')
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = layers_t1.TF2Upstride('basic')(inputs)
    x = layers_t1.Conv2D(32, (3, 3), use_bias=True)(x)
    x = layers_t1.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, TF2Upstride, conv, Upstride2TF
    self.assertEqual(model.count_params(), 1792)
    tf.keras.backend.set_image_data_format('channels_first')

  def test_bias_defined_channels_first(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = layers_t1.TF2Upstride('basic')(inputs)
    x = layers_t1.Conv2D(32, (3, 3), use_bias=True)(x)
    x = layers_t1.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, TF2Upstride, conv, UpStride2TF
    self.assertEqual(model.count_params(), 1792)

  def test_without_bias_channels_last(self):
    tf.keras.backend.set_image_data_format('channels_last')
    inputs = tf.keras.layers.Input((224, 224, 3))
    x = layers_t1.TF2Upstride('basic')(inputs)
    x = layers_t1.Conv2D(32, (3, 3), use_bias=False)(x)
    x = layers_t1.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, TF2Upstride, conv, Upstride2TF
    self.assertEqual(model.count_params(), 1728)
    tf.keras.backend.set_image_data_format('channels_first')

  def test_without_bias_channels_first(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = layers_t1.TF2Upstride('basic')(inputs)
    x = layers_t1.Conv2D(32, (3, 3), use_bias=False)(x)
    x = layers_t1.Upstride2TF()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    self.assertEqual(len(model.layers), 4)  # input, TF2Upstride, conv, UpStride2TF
    self.assertEqual(model.count_params(), 1728)


class TestDropout(unittest.TestCase):
  def test_synchronized(self):
    model = tf.keras.Sequential(layers_t1.Dropout(0.5, synchronized=True, seed=42))
    images = np.ones((10, 10, 10, 3), dtype=np.float32)
    output = model(images, training=True)

    # check that real and imaginary parts are synchronized
    split_output = tf.split(output, 2, axis=0)
    self.assertTrue(np.alltrue((split_output[1] == split_output[0]).numpy()))

  def test_not_synchronized(self):
    model = tf.keras.Sequential(layers_t1.Dropout(0.5, synchronized=False, seed=42))
    images = np.ones((10, 10, 10, 3), dtype=np.float32)
    output = model(images, training=True)

    # check that real and imaginary parts are synchronized
    split_output = tf.split(output, 2, axis=0)
    self.assertFalse(np.alltrue((split_output[1] == split_output[0]).numpy()))


class TestBatchNormalization(unittest.TestCase):
  def test_n_params_channels_first(self):
    """Simple test to count the number of parameters in the batch norm, to be sure we normalize the right axis
    """
    model = tf.keras.Sequential(layers_t1.BatchNormalization(axis=1))
    images = np.ones((10, 3, 10, 10), dtype=np.float32)
    output = model(images, training=True)
    self.assertTrue(model.trainable_weights[0].shape == (6,))  # 3*2 because complex

  def test_n_params_channels_last(self):
    """Simple test to count the number of parameters in the batch norm, to be sure we normalize the right axis
    """
    tf.keras.backend.set_image_data_format('channels_last')
    model = tf.keras.Sequential(layers_t1.BatchNormalization(axis=-1))
    images = np.ones((10, 10, 10, 3), dtype=np.float32)
    output = model(images, training=True)
    self.assertTrue(model.trainable_weights[0].shape == (6,))  # 3*2 because complex

  def test_check_computation(self):
    """This test check that the BN compute the right thing by looking at the mean and variance of the most naive
    possible BN
    """
    tf.keras.backend.set_image_data_format('channels_first')
    bn_layer = layers_t1.BatchNormalization(axis=1, center=False, scale=False, momentum=0)
    model = tf.keras.Sequential(bn_layer)

    # the input of the model is a batch of 2 complex elements with 2 channels. (2* Bs, C) with Bs = 2, C = 2
    # all real parts are equal to 0 and imaginary part to 1
    images = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)

    output = model(images, training=True)

    # the output should be a list of 0
    self.assertTrue(np.alltrue(output.numpy() == np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)))

    # the moving mean should be 0 for real part and 1 for imaginary part
    # the moving variance should be 0 for everything
    for i in range(2):
      if 'moving_mean' in bn_layer.non_trainable_weights[i].name:
        self.assertTrue(np.alltrue(bn_layer.non_trainable_weights[i].numpy() == np.array([0., 0., 1., 1.], dtype=np.float32)))
      if 'moving_variance' in bn_layer.non_trainable_weights[i].name:
        self.assertTrue(np.alltrue(bn_layer.non_trainable_weights[i].numpy() == np.array([0., 0., 0., 0.], dtype=np.float32)))
