import unittest
import numpy as np
import tensorflow as tf
from .initializers import CInitializer, HInitializer, IndependentFilter


class TestCInitializer(unittest.TestCase):
  def test_init(self):
    np.random.seed(50)
    init = CInitializer()
    kernel = init((200, 200))
    kernel_r = kernel[:, :100]
    kernel_i = kernel[:, 100:]

    # for complex, Var(W) = E[|W|**2] (because E[W]**2=0)
    self.assertAlmostEqual(np.mean(kernel_r ** 2 + kernel_i ** 2), 2/(200/2 + 100), 4)


class TestHInitializer(unittest.TestCase):
  def test_init(self):
    np.random.seed(3)
    init = HInitializer()
    kernel = init((200, 400))

    kernel_r, kernel_i, kernel_j, kernel_k = np.split(kernel, 4, axis=1)
    # for quaternion, Var(W) = E[|W|**2] (because E[W]**2=0)
    self.assertAlmostEqual(np.mean(kernel_r ** 2 + kernel_i ** 2 + kernel_j**2 + kernel_k**2), 2/300, 4)  # 300 = 200 (fan_in) + 100 (fan_out)


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
      outputs += self.create_run_check_destroy_neural_net(dense_layer, (BS, 20), 2/(20 + 10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.013664723)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 1.3137447)  # not far from 20 * 2 /(20 + 10) = 4 / 3

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
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 224, 224, 20), 2/(3*3*20 + 10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), -0.00011997313)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 1.9009634)  # not far from 20*3*3 * 2 /(20*3*3 + 10) = 36/19 = 1.894736842105263

  def test_conv_layer_channel_first(self):
    tf.keras.backend.set_image_data_format('channels_first')
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.Conv2D(10, (3, 3), kernel_initializer=IndependentFilter(), use_bias=False, padding='same', data_format='channels_first')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 20, 224, 224), 2/(3*3*20 + 10))
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.0007675069)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 1.9007603, places=6)  # not far from 20*3*3 * 2 /(20*3*3 + 10) = 36/19 = 1.894736842105263
    tf.keras.backend.set_image_data_format('channels_last')

  def test_depthwise_conv_layer(self):
    tf.keras.backend.set_image_data_format('channels_last')
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.DepthwiseConv2D((3, 3), depthwise_initializer=IndependentFilter(depthwise=True), use_bias=False, padding='same')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 224, 224, 20), 2/(3*3 + 1), is_depthwise=True)
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.00046288496)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 1.8047282)  # not far from 3*3 * 2 /(3*3 + 1) = 18/10 = 1.8

  def test_depthwise_conv_layer_channel_first(self):
    tf.keras.backend.set_image_data_format('channels_first')
    N = 10
    BS = 1
    outputs = []
    for i in range(N):
      np.random.seed(i*50)
      conv_layer = tf.keras.layers.DepthwiseConv2D((3, 3), depthwise_initializer=IndependentFilter(depthwise=True), use_bias=False, padding='same', data_format='channels_first')
      outputs += self.create_run_check_destroy_neural_net(conv_layer, (BS, 20, 224, 224), 2/(3*3 + 1), is_depthwise=True)
    print(np.mean(outputs))
    print(np.var(outputs))
    self.assertAlmostEqual(np.mean(outputs), 0.00018445279)  # not far from 0
    self.assertAlmostEqual(np.var(outputs), 1.8032125)  # not far from 3*3 * 2 /(3*3 + 1) = 18/10 = 1.8

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
    self.assertAlmostEqual(np.var(kernel_r), 2/(20 + 10))
    print(np.mean(kernel_i))
    print(np.var(kernel_i))
    self.assertAlmostEqual(np.var(kernel_i), 2/(20 + 10))
