import unittest
import numpy as np
from collections import defaultdict
import tensorflow as tf
from upstride.type1.tf.keras import layers 
from .initializers import CInitializer, HInitializer, IndependentFilter, InitializersFactory
from .test_batchnorm import Channel2Batch, Batch2Channel
from src_test.conv_from_dcn import ComplexConv2D
from .generic_layers import change_upstride_type

class TestCInitializer(unittest.TestCase):
  def test_init(self):
    np.random.seed(50)
    init = CInitializer()
    kernel = init((200, 200))
    kernel_r = kernel[:, :100]
    kernel_i = kernel[:, 100:]

    # for complex, Var(W) = E[|W|**2] (because E[W]**2=0)
    self.assertAlmostEqual(np.mean(kernel_r ** 2 + kernel_i ** 2), 1/(200 + 100), 4)


class TestHInitializer(unittest.TestCase):
  def test_init(self):
    np.random.seed(3)
    init = HInitializer()
    kernel = init((200, 400))

    kernel_r, kernel_i, kernel_j, kernel_k = np.split(kernel, 4, axis=1)
    # for quaternion, Var(W) = E[|W|**2] (because E[W]**2=0)
    self.assertAlmostEqual(np.mean(kernel_r ** 2 + kernel_i ** 2 + kernel_j**2 + kernel_k**2), 1/300, 4)  # 300 = 200 (fan_in) + 100 (fan_out)


class TestIndependentFilter(unittest.TestCase):
  def create_run_check_destroy_neural_net(self, layer, input_shape, kernel_var, is_depthwise=False):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape[1:]),  # 1: is for removing BS
        layer
    ])
    o = model.predict(np.random.normal(0., 1., input_shape))
    # check kernel variance
    if is_depthwise:
      kernel = layer.depthwise_kernel.numpy()
    else:
      kernel = layer.kernel.numpy()
    self.assertAlmostEqual(np.var(kernel), kernel_var)
    tf.keras.backend.clear_session()
    return list(o.flatten())

  def test_dense_layer(self):
    """ test initialization with a dense layer 20 -> 10
    """
    N = 10
    BS = 50
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      dense_layer = tf.keras.layers.Dense(10, kernel_initializer=IndependentFilter(), use_bias=False)
      outputs += self.create_run_check_destroy_neural_net(dense_layer, (BS, 20), 1/(20 + 10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), -0.008028324)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 0.68142286)  # not far from 20 * 1 /(20 + 10) = 2 / 3

  def test_conv_layer(self):
    """ test conv with kernel (3, 3), from 20 channels to 10
    """
    tf.keras.backend.set_image_data_format('channels_last')
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.Conv2D(10, (3, 3), kernel_initializer=IndependentFilter(), use_bias=False, padding='same')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 224, 224, 20), 1/(3*3*20 + 3*3*10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), -7.1164446e-05)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 0.6666, places=2)  # not far from 20*3*3 * 1 /(20*3*3 + 3*3*10) = 2/3
    tf.keras.backend.set_image_data_format('channels_first')

  def test_conv_layer_channel_first(self):
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.Conv2D(10, (3, 3), kernel_initializer=IndependentFilter(), use_bias=False, padding='same')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 20, 224, 224), 1/(3*3*20 + 3*3*10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.00045526333)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 0.666, places=2)  # not far from 20*3*3 * 1 /(20*3*3 + 3*3*10) = 2/3

  def test_depthwise_conv_layer(self):
    tf.keras.backend.set_image_data_format('channels_last')
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.DepthwiseConv2D((3, 3), depthwise_initializer=IndependentFilter(depthwise=True), use_bias=False, padding='same')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 224, 224, 20), 1/(3*3 + 3*3), is_depthwise=True)
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.00024396196)  # not farrom 0
    self.assertAlmostEqual(np.var(outputs), 0.50, places=2)  # not far from 3*3 * 1 /(3*3 + 3*3) = 0.5
    tf.keras.backend.set_image_data_format('channels_first')

  def test_depthwise_conv_layer_channel_first(self):
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.DepthwiseConv2D((3, 3), depthwise_initializer=IndependentFilter(depthwise=True), use_bias=False, padding='same', data_format='channels_first')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 20, 224, 224), 1/(3*3 + 3*3), is_depthwise=True)
    self.assertAlmostEqual(np.mean(outputs), 9.721531e-05)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 0.50, places=2)  # not far from 3*3 * 1 /(3*3 + 3*3) = 0.5

  def test_complex_kernel(self):
    tf.keras.backend.set_image_data_format('channels_last')
    np.random.seed(42)
    # simulate the call of a complex dense layer
    init = IndependentFilter(complex=True)

    # fist call for real part
    kernel = init((20, 20))
    print(kernel.shape)
    kernel_r = kernel[:, :10]
    kernel_i = kernel[:, 10:]
    print(np.mean(kernel_r))
    print(np.var(kernel_r))
    self.assertAlmostEqual(np.var(kernel_r), 1/(20 + 10))
    print(np.mean(kernel_i))
    print(np.var(kernel_i))
    self.assertAlmostEqual(np.var(kernel_i), 1/(20 + 10))
    tf.keras.backend.set_image_data_format('channels_first')

class TestComplexInitCompare(unittest.TestCase):
  def get_model(self, num_of_conv_layer, DCN_ours = True):
    conv_args = {
      "filters": 3,
      "kernel_size": 1,
      "kernel_initializer": "complex_independent" if not DCN_ours else IndependentFilter(criterion='he',complex=True,seed=1337),
      "use_bias": False
    }
    inputs = tf.keras.layers.Input(shape=(6, 3, 3))
    if DCN_ours:
      x = Channel2Batch()(inputs)
      for i in range(num_of_conv_layer):
        x = layers.Conv2D(**conv_args,name=f"conv_{str(i)}")(x)
      x = Batch2Channel()(x)
    else:
      for i in range(num_of_conv_layer):
        x = ComplexConv2D(**conv_args, name=f"conv2d_1/conv_{str(i)}")(inputs)
    x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    return model
    
  def get_statistics(self, num_of_conv_layer, iteration=10, is_dcn_ours=True):
    weight_dict = defaultdict(lambda: defaultdict(list)) # dict(dict(list))
    for i in range(iteration):
      model = self.get_model(num_of_conv_layer, is_dcn_ours)
      for weight in model.weights:
        weight_name = weight.name.split("/")[1] # get the weight name as conv2d_#
        weight_value = weight.numpy()
        weight_dict[weight_name]["mean"].append(np.mean(weight_value)) 
        weight_dict[weight_name]["std"].append(np.std(weight_value))
    return weight_dict

  def test_compare(self):
    # required to force type1 so that the below tests are working. 
    # Need to rethink on how we use global values. This is a a pain to debug.
    change_upstride_type(1,["", "12"],(2, 0, 0))
    weight_dcn_ours = self.get_statistics(1, iteration=3, is_dcn_ours=True)
    weight_dcn_source = self.get_statistics(1, iteration=3, is_dcn_ours=False)
    for (k1, v1), (k2, v2) in zip(sorted(weight_dcn_ours.items()), sorted(weight_dcn_source.items())):
      print(np.mean(v1['mean']), np.mean(v2['mean']))
      print(np.mean(v1['std']), np.mean(v2['std']))
      self.assertTrue(np.allclose(np.mean(v1['mean']),np.mean(v2['mean'])))
      self.assertTrue(np.allclose(np.mean(v1['std']),np.mean(v2['std'])))
