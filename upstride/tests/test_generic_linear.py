import pytest
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.conv_utils import convert_data_format
from functools import lru_cache
from upstride import generic_layers
from upstride import convolutional
from upstride.uptypes_utilities import UPTYPE0, UPTYPE1, UPTYPE2
from .utility import gpu_visible, assert_small_float_difference, random_float_tensor, random_integer_tensor


### Tests infrastructure

class GenericTestBase:

    uptypes = {
        'up0' : UPTYPE0,
        'up1' : UPTYPE1,
        'up2' : UPTYPE2,
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
            [(1, 1), (0, -1), (3, -1), (2, 1)],
            [(2, 1), (3, 1), (0, -1), (1, -1)],
            [(3, 1), (2, -1), (1, 1), (0, -1)],
        ]),
    }

    def setup(self, random_tensor=random_integer_tensor):
        self.random_tensor = random_tensor

    # assumes zero-filled / no bias
    def generic_linear_test(self, layer_test, layer_ref, uptype, component_shape):
        algebra_map = self.algebra_maps[uptype]
        hyper_dimension = self.uptypes[uptype].multivector_length

        # prepare input hyper-complex components
        components = []
        for _ in range(hyper_dimension):
            component = self.random_tensor(component_shape)
            components.append(component)

        # run test operation
        inp = tf.concat(components, axis=0)
        test_out = layer_test(inp)

        # prepare filter weights for the reference operation
        w = layer_test.get_weights()[0]
        if len(w.shape) != 5:
            w = [w[..., i::hyper_dimension] for i in range(hyper_dimension)]

        # run reference once to initialize weights
        layer_ref(components[0])
        zero_bias = tf.zeros_like(layer_ref.get_weights()[1]) if layer_ref.use_bias else None

        # compute intermediate component x component results
        ref_partial = [[] for _ in range(hyper_dimension)]
        for i in range(hyper_dimension):
            layer_ref_weights = [w[i], zero_bias] if layer_ref.use_bias else [w[i]]
            layer_ref.set_weights(layer_ref_weights)
            for j in range(hyper_dimension):
                inter_res = layer_ref(components[j])
                ref_partial[i].append(inter_res)

        # sum intermediate results according to the algebra
        ref_outputs = [0 for _ in range(hyper_dimension)]
        for i in range(hyper_dimension):
            for j in range(hyper_dimension):
                which, coeff = algebra_map[i][j]
                ref_outputs[which] = ref_outputs[which] + ref_partial[i][j] * coeff

        # add bias if necessary
        if layer_ref.use_bias:
            bias = layer_test.get_weights()[1]
            if len(bias.shape) != 2:
                bias = [bias[i::hyper_dimension] for i in range(hyper_dimension)]
            for i in range(hyper_dimension):
                ref_outputs[i] = self.add_bias_component(ref_outputs[i], bias[i], layer_ref, component_shape)

        # verify if results match
        ref_out = tf.concat(ref_outputs, axis=0)
        assert_small_float_difference(test_out, ref_out, 0.001)

    @lru_cache(maxsize=4)
    def get_op_data_format(self, tensor_shape):
        return convert_data_format(tf.keras.backend.image_data_format().lower(), len(tensor_shape))

    def add_bias_component(self, tensor, bias, layer_ref, component_shape):
        return tf.add(tensor, bias) if 'Dense' in str(layer_ref) else tf.nn.bias_add(tensor, bias, self.get_op_data_format(component_shape))

    def assert_right_kernel_size(self, component_shape, raise_error=True, **kwargs):
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

    def try_set_filter_initializer(self, layer_test_cls, **kwargs):
        if 'Depthwise' in str(layer_test_cls) and 'depthwise_initializer' not in kwargs:
            kwargs['depthwise_initializer'] = self.random_tensor
        elif 'kernel_initializer' not in kwargs:
            kwargs['kernel_initializer'] = self.random_tensor
        return kwargs

    def try_set_bias(self, **kwargs):
        if 'use_bias' in kwargs:
            if kwargs['use_bias'] and 'bias_initializer' not in kwargs:
                kwargs['bias_initializer'] = self.random_tensor
        else:
            kwargs['use_bias'] = False
        return kwargs

    def layers_test(self, component_shape, uptype, layer_test_cls, layer_ref_cls, **kwargs):
        kwargs = self.try_set_filter_initializer(layer_test_cls, **kwargs)
        kwargs = self.try_set_bias(**kwargs)

        layer_test = layer_test_cls(self.uptypes[uptype], **kwargs)
        layer_ref = layer_ref_cls(**kwargs)
        self.generic_linear_test(layer_test, layer_ref, uptype, component_shape)

    def convolution_test(self, channel_convention, component_shape, uptype, layer_test_cls, layer_ref_cls, **kwargs):
        if not self.assert_right_kernel_size(component_shape, **kwargs):
            return

        tf.keras.backend.set_image_data_format(channel_convention)
        nhwc_to_nchw_perm = (0, 3, 1, 2)
        if channel_convention == 'channels_first':
            component_shape = tuple(component_shape[i] for i in nhwc_to_nchw_perm)

        self.layers_test(component_shape, uptype, layer_test_cls, layer_ref_cls, **kwargs)


### Test configurations

class TestGenericLinearSimple(GenericTestBase):

    def test_simple_Conv2D(self):
        self.convolution_test('channels_last', (2, 3, 3, 2), 'up1', generic_layers.Conv2D, tf.keras.layers.Conv2D, filters=2, kernel_size=3)

    def test_simple_DepthwiseConv2D(self):
        self.convolution_test('channels_last', (2, 3, 3, 2), 'up1', generic_layers.DepthwiseConv2D, tf.keras.layers.DepthwiseConv2D, kernel_size=3)

    def test_simple_Dense(self):
        self.layers_test((2, 2), 'up1', generic_layers.Dense, tf.keras.layers.Dense, units=2)


@pytest.mark.exhaustive
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestGenericLinearExhaustive(GenericTestBase):

    standard_params = [3, 5, 16]

    def setup(self):
        super().setup(random_float_tensor)

    @pytest.mark.parametrize('units', standard_params)
    @pytest.mark.parametrize('batch_size', standard_params)
    @pytest.mark.parametrize('channels', standard_params)
    def test_Dense(self, units, batch_size, channels, uptype):
        kwargs = {
            'units' : units,
        }
        component_shape = (batch_size, channels)
        self.layers_test(component_shape, uptype, generic_layers.Dense, tf.keras.layers.Dense, **kwargs)


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
        self.convolution_test(channel_convention, component_shape, uptype, generic_layers.Conv2D, tf.keras.layers.Conv2D, **kwargs)


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
        self.convolution_test(channel_convention, component_shape, uptype, generic_layers.DepthwiseConv2D, tf.keras.layers.DepthwiseConv2D, **kwargs)


@pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestConv2D(GenericTestBase):

    def run_test(self, channel_convention, component_shape, uptype, **kwargs):
        self.convolution_test(channel_convention, component_shape, uptype, generic_layers.Conv2D, tf.keras.layers.Conv2D, **kwargs)

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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
            'use_bias' : True,
        }
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


class TestConv2DParcollet(TestConv2D):

    def run_test(self, channel_convention, component_shape, uptype, **kwargs):
        self.convolution_test(channel_convention, component_shape, uptype, convolutional.Conv2DParcollet, tf.keras.layers.Conv2D, **kwargs)


@pytest.mark.parametrize('channel_convention', ['channels_first', 'channels_last'])
@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestDepthwiseConv2D(GenericTestBase):

    def run_test(self, channel_convention, component_shape, uptype, **kwargs):
        self.convolution_test(channel_convention, component_shape, uptype, generic_layers.DepthwiseConv2D, tf.keras.layers.DepthwiseConv2D, **kwargs)

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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (7, 5, 5, 8),
        (9, 10, 10, 4),
        (16, 3, 3, 1),
    ])
    @pytest.mark.parametrize('kernel_size', [1, 3])
    def test_bias(self, component_shape, kernel_size, channel_convention, uptype):
        kwargs = {
            'kernel_size' : kernel_size,
            'use_bias' : True,
        }
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


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
        self.run_test(channel_convention, component_shape, uptype, **kwargs)


@pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
class TestDense(GenericTestBase):

    def run_test(self, component_shape, uptype, **kwargs):
        self.layers_test(component_shape, uptype, generic_layers.Dense, tf.keras.layers.Dense, **kwargs)

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
        self.run_test(component_shape, uptype, **kwargs)


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
        self.run_test(component_shape, uptype, **kwargs)


    @pytest.mark.parametrize('component_shape', [
        (1, 1),
        (19, 4),
        (29, 31),
    ])
    @pytest.mark.parametrize('units', [1, 17, 32])
    def test_bias(self, component_shape, units, uptype):
        kwargs = {
            'units' : units,
            'use_bias' : True,
        }
        self.run_test(component_shape, uptype, **kwargs)
