import pytest
import tensorflow as tf
import numpy as np
from upstride import generic_layers
from collections import namedtuple


### Tests infrastructure

UpType = namedtuple('UpType', ['upstride_type', 'blade_indexes', 'geometrical_def'])

uptypes = {
    'up0' : UpType(0, [''], (0, 0, 0)),
    'up1' : UpType(1, ['', '12'], (2, 0, 0)),
    'up2' : UpType(2, ['', '12', '13', '23'], (3, 0, 0)),
}

algebra_maps = {
    'up0' : np.array([
        [(0, 1)],
    ]),
    'up1' : np.array([
        [(0, 1), (1, 1)],
        [(1, 1), (0, -1)],
    ]),
    'up2' : np.array([
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(1, 1), (0, -1), (3, 1), (2, -1)],
        [(2, 1), (3, -1), (0, -1), (1, 1)],
        [(3, 1), (2, 1), (1, -1), (0, -1)],
    ]),
}


def gpu_visible():
  """ Returns True if TF sees GPU
  """
  return tf.config.list_physical_devices('GPU') != []


def assert_small_float_difference(tensor1, tensor2, relative_error_threshold):
    """ Asserts float tensors differ by no more than threshold scaled by the values checked
    """
    abs_diff = tf.abs(tensor1 - tensor2)
    abs_max_tensors = tf.abs(tf.maximum(tensor1, tensor2))
    threshold = relative_error_threshold * (1 + abs_max_tensors)
    assert tf.reduce_all(abs_diff < threshold)


# assumes zero-filled / no bias
def generic_linear_test(layer_test, layer_ref, uptype, component_shape):
    algebra_map = algebra_maps[uptype]
    hyper_dimension = len(algebra_map)

    components = []
    for _ in range(hyper_dimension):
        component = np.random.rand(*component_shape).astype('f')
        components.append(component)

    inp = np.concatenate(components)
    test_out = layer_test(inp)

    w = layer_test.get_weights()[0]
    bias = layer_test.get_weights()[1]
    w_components = np.split(w, hyper_dimension, axis=-1)

    layer_ref(components[0])
    zero_bias = np.zeros_like(layer_ref.get_weights()[1])

    ref_partial = [[] for _ in range(hyper_dimension)]
    for i in range(hyper_dimension):
        layer_ref.set_weights([w_components[i], zero_bias])
        for j in range(hyper_dimension):
            inter_res = layer_ref(components[j])
            ref_partial[i].append(inter_res)

    ref_outputs = [0 for _ in range(hyper_dimension)]
    for i in range(hyper_dimension):
        for j in range(hyper_dimension):
            which, coeff = algebra_map[i][j]
            ref_outputs[which] = ref_outputs[which] + ref_partial[i][j] * coeff

    # print(bias.shape)
    # print(layer_test.get_weights()[0].shape)

    # for i in range(hyper_dimension):
        # ref_outputs[i] = tf.nn.bias_add(ref_outputs[i], bias[i])

    ref_out = np.concatenate(ref_outputs)

    assert_small_float_difference(test_out, ref_out, 0.0001)


def assert_right_kernel_size(component_shape, raise_error=True, **kwargs):
    _, height, width, _ = component_shape
    kernel_size = kwargs['kernel_size']
    if type(kernel_size) == int:
        if kernel_size > min(height, width):
            if raise_error:
                raise ValueError(f'Kernel size ({kernel_size}) too large for the image size ({height} x {width})')
            return False
    elif type(kernel_size) == tuple and len(kernel_size) == 2:
        if kernel_size[0] > height or kernel_size[1] > width:
            if raise_error:
                raise ValueError(f'Kernel size ({kernel_size}) too large for the image size ({height} x {width})')
            return False
    else:
        raise ValueError(f'Kernel size in incorrect format: {kernel_size}')
    return True


def layers_test(component_shape, uptype, layer_test_cls, layer_ref_cls, **kwargs):
    layer_test = layer_test_cls(*uptypes[uptype], **kwargs)
    layer_ref = layer_ref_cls(**kwargs)
    generic_linear_test(layer_test, layer_ref, uptype, component_shape)


def convolution_test(channel_convention, component_shape, uptype, layer_test_cls, layer_ref_cls, **kwargs):
    if not assert_right_kernel_size(component_shape, **kwargs):
        return

    tf.keras.backend.set_image_data_format(channel_convention)
    nhwc_to_nchw_perm = (0, 3, 1, 2)
    if channel_convention == 'channels_first':
        component_shape = tuple(component_shape[i] for i in nhwc_to_nchw_perm)

    layers_test(component_shape, uptype, uptype, layer_test_cls, layer_ref_cls, **kwargs)


### Test configurations

@pytest.mark.exhaustive
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestGenericLinearExhaustive:

    standard_params = [3, 5, 16]

    @pytest.mark.parametrize('units', standard_params)
    @pytest.mark.parametrize('batch_size', standard_params)
    @pytest.mark.parametrize('channels', standard_params)
    def test_Dense(self, units, batch_size, channels, uptype):
        kwargs = {
            'units' : units,
        }
        component_shape = (batch_size, channels)
        layers_test(component_shape, uptype, generic_layers.Dense, tf.keras.layers.Dense, **kwargs)


    @pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
    @pytest.mark.parametrize('filters', standard_params)
    @pytest.mark.parametrize('kernel_size', [1, 3])
    @pytest.mark.parametrize('height', standard_params)
    @pytest.mark.parametrize('width', standard_params)
    @pytest.mark.parametrize('batch_size', standard_params)
    @pytest.mark.parametrize('channels', standard_params)
    def test_Conv2D(self, channel_convention, filters, kernel_size, height, width, batch_size, channels, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : kernel_size,
        }
        component_shape = (batch_size, height, width, channels)
        convolution_test(channel_convention, component_shape, uptype, generic_layers.Conv2D, tf.keras.layers.Conv2D, **kwargs)


    @pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    @pytest.mark.parametrize('height', standard_params)
    @pytest.mark.parametrize('width', standard_params)
    @pytest.mark.parametrize('batch_size', standard_params)
    @pytest.mark.parametrize('channels', standard_params)
    def test_DepthwiseConv2D(self, channel_convention, kernel_size, height, width, batch_size, channels, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
        }
        component_shape = (batch_size, height, width, channels)
        convolution_test(channel_convention, component_shape, uptype, generic_layers.DepthwiseConv2D, tf.keras.layers.DepthwiseConv2D, **kwargs)


@pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestConv2D:

    layer_test_cls = generic_layers.Conv2D
    layer_ref_cls = tf.keras.layers.Conv2D

    @pytest.mark.parametrize('component_shape', [
        (1, 5, 5, 8),
        (1, 10, 10, 4),
        (1, 3, 3, 1),
    ])
    @pytest.mark.parametrize('filters', [3, 8, 32])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_basic(self, component_shape, filters, kernel_size, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 1),
    ])
    @pytest.mark.parametrize('filters', [1, 9, 32])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_bigger_batch(self, component_shape, filters, kernel_size, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 6, 6, 8),
        (9, 11, 11, 4),
        (16, 2, 2, 1),
    ])
    @pytest.mark.parametrize('filters', [1, 15, 32])
    def test_pointwise(self, component_shape, filters, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 1,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (5, 7, 7, 8),
        (10, 11, 11, 4),
        (14, 9, 9, 1),
    ])
    @pytest.mark.parametrize('filters', [8, 13, 32])
    @pytest.mark.parametrize('strides', [
        (2, 2),
        (2, 3),
        (3, 1)
    ])
    def test_strided(self, component_shape, filters, strides, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 3,
            'strides' : strides,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (5, 7, 7, 4),
        (10, 11, 11, 5),
        (14, 9, 9, 2),
    ])
    @pytest.mark.parametrize('filters', [4, 19, 32])
    @pytest.mark.parametrize('dilation_rate', [
        (2, 2),
        (2, 3),
        (3, 1)
    ])
    def test_dilated(self, component_shape, filters, dilation_rate, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 3,
            'dilation_rate' : dilation_rate,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 9, 8),
        (9, 5, 10, 4),
        (16, 9, 3, 1),
    ])
    @pytest.mark.parametrize('filters', [2, 11, 32])
    @pytest.mark.parametrize('kernel_size', [
        2,
        (1, 3),
        (3, 2),
    ])
    def test_non_square(self, component_shape, filters, kernel_size, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 1),
    ])
    @pytest.mark.parametrize('filters', [1, 9, 32])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_bias(self, component_shape, filters, kernel_size, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : kernel_size,
            # 'use_bias' : True,
            # 'bias_initializer' : 'glorot_uniform',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (9, 5, 5, 7),
        (4, 3, 3, 6),
        (14, 9, 9, 10),
    ])
    @pytest.mark.parametrize('filters', [5, 18, 32])
    def test_padded(self, component_shape, filters, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 3,
            'padding' : 'same',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (9, 5, 9, 7),
        (8, 5, 10, 3),
        (14, 9, 3, 9),
    ])
    @pytest.mark.parametrize('filters', [3, 15, 32])
    def test_assymetrically_padded(self, component_shape, filters, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 3,
            'padding' : 'same',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (11, 9, 9, 3),
        (5, 8, 8, 8),
        (8, 7, 7, 13),
    ])
    @pytest.mark.parametrize('padding', ['same', 'valid'])
    @pytest.mark.parametrize('strides', [
        (2, 3),
        (3, 3),
        (3, 2),
    ])
    def test_padded_strided(self, component_shape, padding, strides, channel_convention, uptype):
        kwargs = {
            'filters' : 32,
            'kernel_size' : 3,
            'padding' : padding,
            'strides' : strides,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (11, 9, 9, 3),
        (5, 8, 8, 8),
        (8, 7, 7, 13),
    ])
    @pytest.mark.parametrize('padding', ['same', 'valid'])
    @pytest.mark.parametrize('dilation_rate', [
        (2, 3),
        (3, 3),
        (3, 2),
    ])
    def test_padded_dilated(self, component_shape, padding, dilation_rate, channel_convention, uptype):
        kwargs = {
            'filters' : 32,
            'kernel_size' : 3,
            'padding' : padding,
            'dilation_rate' : dilation_rate,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.skipif(not gpu_visible(), reason="grouped conv not supported on CPU")
    @pytest.mark.skipif(tf.version.VERSION < '2.3.0', reason="tensorflow version needs to be at least 2.3.0")
    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 20),
    ])
    @pytest.mark.parametrize('filters', [4, 8, 32])
    @pytest.mark.parametrize('groups', [2, 4])
    def test_grouped(self, component_shape, filters, groups, channel_convention, uptype):
        kwargs = {
            'filters' : filters,
            'kernel_size' : 3,
            'groups' : groups,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


@pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestDepthwiseConv2D:

    layer_test_cls = generic_layers.DepthwiseConv2D
    layer_ref_cls = tf.keras.layers.DepthwiseConv2D

    @pytest.mark.parametrize('component_shape', [
        (1, 5, 5, 8),
        (1, 10, 10, 4),
        (1, 3, 3, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_basic(self, component_shape, kernel_size, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_bigger_batch(self, component_shape, kernel_size, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 6, 6, 8),
        (9, 11, 11, 4),
        (16, 4, 4, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    @pytest.mark.parametrize('depth_multiplier', [3, 9, 16])
    def test_depth_multiplier(self, component_shape, kernel_size, depth_multiplier, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
            'depth_multiplier' : depth_multiplier,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (5, 7, 7, 8),
        (10, 11, 11, 4),
        (14, 9, 9, 1),
    ])
    @pytest.mark.parametrize('strides', [
        (2, 2),
        (3, 3),
    ])
    def test_strided(self, component_shape, strides, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'strides' : strides,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (5, 7, 7, 4),
        (10, 11, 11, 5),
        (14, 9, 9, 2),
    ])
    @pytest.mark.parametrize('dilation_rate', [
        (2, 2),
        (2, 3),
        (3, 1)
    ])
    def test_dilated(self, component_shape, dilation_rate, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'dilation_rate' : dilation_rate,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 9, 8),
        (9, 5, 10, 4),
        (16, 9, 3, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [
        2,
        (1, 3),
        (3, 2),
    ])
    def test_non_square(self, component_shape, kernel_size, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_bias(self, component_shape, kernel_size, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
            # 'use_bias' : True,
            # 'bias_initializer' : 'glorot_uniform',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (9, 5, 5, 7),
        (4, 3, 3, 6),
        (14, 9, 9, 10),
    ])
    def test_padded(self, component_shape, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'padding' : 'same',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (9, 5, 9, 7),
        (8, 5, 10, 3),
        (14, 9, 3, 9),
    ])
    def test_assymetrically_padded(self, component_shape, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'padding' : 'same',
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (11, 9, 9, 3),
        (5, 8, 8, 8),
        (8, 7, 7, 13),
    ])
    @pytest.mark.parametrize('padding', ['same', 'valid'])
    @pytest.mark.parametrize('strides', [
        (2, 2),
        (3, 3),
    ])
    def test_padded_strided(self, component_shape, padding, strides, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'padding' : padding,
            'strides' : strides,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (11, 9, 9, 3),
        (5, 8, 8, 8),
        (8, 7, 7, 13),
    ])
    @pytest.mark.parametrize('padding', ['same', 'valid'])
    @pytest.mark.parametrize('dilation_rate', [
        (2, 3),
        (3, 3),
        (3, 2),
    ])
    def test_padded_dilated(self, component_shape, padding, dilation_rate, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'padding' : padding,
            'dilation_rate' : dilation_rate,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.skipif(not gpu_visible(), reason="grouped conv not supported on CPU")
    @pytest.mark.skipif(tf.version.VERSION < '2.3.0', reason="tensorflow version needs to be at least 2.3.0")
    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 20),
    ])
    @pytest.mark.parametrize('groups', [2, 4])
    def test_grouped(self, component_shape, groups, channel_convention, uptype):
        kwargs = {
            'kernel_size' : 3,
            'groups' : groups,
        }
        convolution_test(channel_convention, component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestDense:

    layer_test_cls = generic_layers.Dense
    layer_ref_cls = tf.keras.layers.Dense

    @pytest.mark.parametrize('component_shape', [
        (1, 8),
        (1, 24),
        (1, 1),
    ])
    @pytest.mark.parametrize('units', [1, 7, 16])
    def test_basic(self, component_shape, units, uptype):
        kwargs = {
            'units' : units,
        }
        layers_test(component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (9, 4),
        (16, 24),
        (23, 15),
    ])
    @pytest.mark.parametrize('units', [1, 17, 32])
    def test_bigger_batch(self, component_shape, units, uptype):
        kwargs = {
            'units' : units,
        }
        layers_test(component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (6, 12),
        (19, 4),
        (29, 31),
    ])
    @pytest.mark.parametrize('units', [1, 17, 32])
    def test_bias(self, component_shape, units, uptype):
        kwargs = {
            'units' : units,
            # 'use_bias' : True,
            # 'bias_initializer' : 'glorot_uniform',
        }
        layers_test(component_shape, uptype, self.layer_test_cls, self.layer_ref_cls, **kwargs)