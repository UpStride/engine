import unittest
import pytest
import numpy as np
import tensorflow as tf
from upstride.generic_layers import ga_multiply_get_index, unit_multiplier, square_vector
from upstride.uptypes_utilities import UPTYPE0, UPTYPE1, UPTYPE2, UPTYPE3, UpstrideDatatype
from upstride.tests.utility import random_integer_tensor, assert_small_float_difference
from upstride import generic_layers
from upstride.type1.tf.keras import layers as layers_t1
from .test_generic_linear import GenericTestBase


UPTYPE2_alternative = UpstrideDatatype(2, (0, 2, 0), ['', '1', '2', '12'])

uptypes = {
  'up0' : UPTYPE0,
  'up1' : UPTYPE1,
  'up2' : UPTYPE2,
  'up3' : UPTYPE3,
  'up2_alt' : UPTYPE2_alternative,
}


def ga_multiply_get_index_by_insertion_sort(uptype, index_1, index_2):
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


class TestGAUtilities:

  @pytest.mark.parametrize('uptype, ref_squares', [
    ('up0', []),
    ('up1', [1] * 2),
    ('up2', [1] * 3),
    ('up3', [1] * 3),
    ('up2_alt', [-1] * 2)
  ])
  def test_squaring(self, uptype, ref_squares):
    uptype = uptypes[uptype]
    base_indexes = sum(uptype.geometrical_def)
    test_squares = [square_vector(uptype, i) for i in range(1, base_indexes + 1)]
    assert test_squares == ref_squares

  @pytest.mark.parametrize('uptype, index_1, index_2', [
    ('up1', '1', '12'),
    ('up1', '12', '12'),
    ('up2', '23', '13'),
    ('up2', '13', '12'),
    ('up3', '123', '1'),
    ('up3', '123', '23'),
    ('up2_alt', '2', '1'),
    ('up2_alt', '12', '1'),
  ])
  def test_ga_multiply_get_index(self, uptype, index_1, index_2):
    uptype = uptypes[uptype]
    assert ga_multiply_get_index(uptype, index_1, index_2) == ga_multiply_get_index_by_sort(uptype, index_1, index_2)

  @pytest.mark.parametrize('uptype, i, j', [
    ('up1', 1, 0),
    ('up2', 3, 1),
    ('up3', 5, 6),
    ('up2_alt', 2, 3),
  ])
  def test_unit_multiplier(self, uptype, i, j):
    uptype = uptypes[uptype]
    test_mult = unit_multiplier(uptype, i, j)
    ref_sign, ref_index = ga_multiply_get_index(uptype, uptype.blade_indexes[i], uptype.blade_indexes[j])
    assert test_mult == (uptype.blade_indexes.index(ref_index), ref_sign)



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


@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
@pytest.mark.parametrize('component_shape', [
  (5, 4, 7),
  (3, 8, 9),
  (11, 3, 2),
])
@pytest.mark.parametrize('channel_convention', ['channels_last', 'channels_first'])
class TestBatchNormalizationCompute:

  def preparation(self, uptype, component_shape, channel_convention):
    axis = 1 if channel_convention == 'channels_first' else -1
    hyper_dimension = uptypes[uptype].multivector_length
    tf.keras.backend.set_image_data_format(channel_convention)
    nic_to_nci_perm = (0, 2, 1)
    if channel_convention == 'channels_first':
      component_shape = tuple(component_shape[i] for i in nic_to_nci_perm)

    components = []
    for _ in range(hyper_dimension):
      component = random_integer_tensor(component_shape)
      components.append(component)

    return components, axis, hyper_dimension

  def compute_upstride_bn(self, uptype, axis, components):
    kwargs = {
      'gamma_initializer' : random_integer_tensor,
      'beta_initializer' : random_integer_tensor,
      'epsilon' : 0,
    }

    layer = generic_layers.BatchNormalization(uptypes[uptype], axis=axis, **kwargs)

    inp = tf.concat(components, axis=0)
    test_out = layer(inp, training=True)

    return test_out, layer.get_weights()

  def test_manual(self, uptype, component_shape, channel_convention):
    components, axis, hyper_dimension = self.preparation(uptype, component_shape, channel_convention)
    test_out, bn_weights = self.compute_upstride_bn(uptype, axis, components)
    gamma, beta, moving_mean, moving_variance = list(map(lambda weight : tf.split(weight, hyper_dimension), bn_weights))

    normalize = lambda x : (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
    channels = components[0].shape[axis]
    ref_out_unscaled = [tf.concat(list(map(normalize, tf.split(comp, channels, axis=axis))), axis=axis) for comp in components]

    if channel_convention == 'channels_first':
      ref_out_unscaled = [tf.transpose(x, (0, 2, 1)) for x in ref_out_unscaled]

    ref_out = tf.concat([ref_out_unscaled[i] * gamma[i] + beta[i] for i in range(hyper_dimension)], axis=0)

    if channel_convention == 'channels_first':
      ref_out = tf.transpose(ref_out, (0, 2, 1))

    assert_small_float_difference(test_out, ref_out, 0.001)

  def test_basic(self, uptype, component_shape, channel_convention):
    components, axis, hyper_dimension = self.preparation(uptype, component_shape, channel_convention)
    test_out, bn_weights = self.compute_upstride_bn(uptype, axis, components)
    gamma, beta, moving_mean, moving_variance = list(map(lambda weight : tf.split(weight, hyper_dimension), bn_weights))

    layer_ref = tf.keras.layers.BatchNormalization(axis=axis, epsilon=0)
    layer_ref(components[0])
    ref_outs = []
    for i, comp in enumerate(components):
        layer_ref.set_weights([gamma[i], beta[i], moving_mean[i], moving_variance[i]])
        ref_outs.append(layer_ref(comp, training=True))

    ref_out = tf.concat(ref_outs, axis=0)

    assert_small_float_difference(test_out, ref_out, 0.001)

  def test_correct_statistics(self, uptype, component_shape, channel_convention):
    components, axis, hyper_dimension = self.preparation(uptype, component_shape, channel_convention)
    test_out, bn_weights = self.compute_upstride_bn(uptype, axis, components)
    gamma, beta, _, _ = bn_weights

    test_out_comps = tf.split(test_out, hyper_dimension, axis=0)
    reduction_dims = [0,2] if channel_convention == 'channels_first' else [0,1]

    test_out_gamma = tf.concat([tf.math.reduce_std(comp, reduction_dims) for comp in test_out_comps], axis=0)
    test_out_beta = tf.concat([tf.reduce_mean(comp, reduction_dims) for comp in test_out_comps], axis=0)

    assert_small_float_difference(test_out_gamma, tf.abs(gamma))
    assert_small_float_difference(test_out_beta, beta)


@pytest.mark.parametrize('channel_convention', ['channels_last', 'channels_first'])
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestTF2Upstride(GenericTestBase):

    def run_test(self, channel_convention, component_shape, uptype, **kwargs):
        self.layers_test(channel_convention, component_shape, uptype, generic_layers.TF2Upstride, **kwargs)

    @pytest.mark.parametrize('component_shape', [
        (1, 5, 5, 8),
        (1, 10, 10, 4),
        (1, 3, 3, 1),
    ])
    @pytest.mark.parametrize('strategy', ['learned', 'basic', '', 'grayscale', 'joint'])
    def test_basic(self, component_shape, strategy, channel_convention, uptype):
        kwargs = {
            'strategy' : strategy,
        }
        if uptype == uptypes['up2'] or strategy not in ['grayscale', 'joint']:
          self.run_test(channel_convention, component_shape, uptype, **kwargs)

    def layers_test(self, channel_convention, component_shape, uptype, layer_test_cls, **kwargs):
        tf.keras.backend.set_image_data_format(channel_convention)
        nhwc_to_nchw_perm = (0, 3, 1, 2)
        if channel_convention == 'channels_first':
            component_shape = tuple(component_shape[i] for i in nhwc_to_nchw_perm)

        inputs = self.random_tensor(component_shape)

        layer_test = layer_test_cls(self.uptypes[uptype], **kwargs)
        output = layer_test(inputs)
        assert output.shape[0] == inputs.shape[0] * self.uptypes[uptype].multivector_length
        assert output.shape[1:] == inputs.shape[1:]
        if kwargs.get('strategy') == 'basic' or '':
          assert tf.reduce_sum(output[inputs.shape[0]:, ...]) == 0
        elif kwargs.get('strategy') == 'grayscale':
          assert tf.reduce_sum(output[:inputs.shape[0], ...]) == 0
