import unittest
import tensorflow as tf
from upstride.generic_layers import GenericLinear, ga_multiply_get_index, unit_multiplier, UPTYPE3
import numpy as np
from upstride import generic_layers
from upstride.type1.tf.keras import layers as layers_t1


def ga_multiply_get_index_by_insertion_sort(uptype, index_1, index_2):
  """ Given \beta_{index_1}, \beta_{index_2} return (s, index) such as \beta_{index_1} * \beta_{index_2} = s * \beta_{index}
  """
  l = [int(i) for i in index_1] + [int(j) for j in index_2] + ['guard'] # adding 'guard' to the end of the list for easier pairs removal

  n = len(l) - 1
  inversions = 0
  head = 1
  # insertion sort with inversions count, invariant: l[:head] already sorted
  while head < n:
    i = head
    while i > 0 and l[i-1] > l[i]:
      l[i-1], l[i] = l[i], l[i-1]
      inversions += 1
      i -= 1
    head += 1

  out_l = []
  s = 1
  i = 0
  # remove paired indexes
  while i < n:
    if l[i] != l[i+1]: # can check l[i+1] thanks to the 'guard' element
      out_l.append(l[i])
    else:
      s *= square_vector(uptype, l[i])
      i += 1
    i += 1

  return s * (-1 if inversions % 2 == 1 else 1), "".join([str(i) for i in out_l])


def ga_multiply_get_index_by_sort(uptype, index_1, index_2):
  """ Given \beta_{index_1}, \beta_{index_2} return (s, index) such as \beta_{index_1} * \beta_{index_2} = s * \beta_{index}
  """
  l = [int(i) for i in index_1] + [int(j) for j in index_2]
  n = len(l)
  inversions = 0
  # count inversions
  for i in range(n):
    for j in range(i+1, n):
      if l[i] > l[j]:
        inversions += 1

  l = ['x'] + sorted(l) + ['y'] # add guards at both ends for easier indexing
  out_l = []
  s = 1
  # remove paired indexes
  for i in range(1, n+1):
    if l[i-1] == l[i]:
      s *= square_vector(uptype, l[i])
    elif l[i] == l[i+1]:
      pass
    else:
      out_l.append(l[i])

  return s * (-1 if inversions % 2 == 1 else 1), "".join([str(i) for i in out_l])


class TestGAMultiplication(unittest.TestCase):
  def test_ga_multiply_get_index(self):
    s, index = ga_multiply_get_index(UPTYPE3, "123", "12")
    self.assertEqual(s, -1)
    self.assertEqual(index, "3")
    s, index = ga_multiply_get_index(UPTYPE3, "13", "12")
    self.assertEqual(s, 1)
    self.assertEqual(index, "23")
    s, index = ga_multiply_get_index(UPTYPE3, "3", "2")
    self.assertEqual(s, -1)
    self.assertEqual(index, "23")
    s, index = ga_multiply_get_index(UPTYPE3, "32", "32")
    self.assertEqual(s, -1)
    self.assertEqual(index, "")
    s, index = ga_multiply_get_index(UPTYPE3, "2", "2")
    self.assertEqual(s, 1)
    self.assertEqual(index, "")

  def test_unit_multiplier(self):
    # order : (scalar, e1, e2, e3, e12, e13, e23, e123)
    self.assertEqual(unit_multiplier(UPTYPE3, 0, 0), (0, 1))  # 1*1 = 1
    self.assertEqual(unit_multiplier(UPTYPE3, 3, 3), (0, 1))  # e_3*e_3 = 1
    self.assertEqual(unit_multiplier(UPTYPE3, 4, 5), (6, -1))  # e12 * e13 = -e23
    self.assertEqual(unit_multiplier(UPTYPE3, 6, 7), (1, -1))  # e23 * e123 = -e1

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