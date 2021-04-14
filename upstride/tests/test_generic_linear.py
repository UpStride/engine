import pytest
import tensorflow as tf
import numpy as np
from upstride import generic_layers
from collections import namedtuple


UpType = namedtuple('UpType', ['upstride_type', 'blade_indexes', 'geometrical_def'])

uptypes = {
    0 : UpType(0, [''], (0, 0, 0)),
    1 : UpType(1, ['', '12'], (2, 0, 0)),
    2 : UpType(2, ['', '12', '13', '23'], (3, 0, 0)),
}

algebra_maps = {
    0 : np.array([
        [(0, 1)],
    ]),
    1 : np.array([
        [(0, 1), (1, 1)],
        [(1, 1), (0, -1)],
    ]),
    2 : np.array([
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(1, 1), (0, -1), (3, 1), (2, -1)],
        [(2, 1), (3, -1), (0, -1), (1, 1)],
        [(3, 1), (2, 1), (1, -1), (0, -1)],
    ]),
}


def assert_small_float_difference(tensor1, tensor2, relative_error_threshold):
    """ Asserts float tensors differ by no more than threshold scaled by the values checked
    """
    abs_diff = tf.abs(tensor1 - tensor2)
    abs_max_tensors = tf.abs(tf.maximum(tensor1, tensor2))
    threshold = relative_error_threshold * (1 + abs_max_tensors)
    assert tf.reduce_all(abs_diff < threshold)


# assumes zero-filled / no bias
def generic_linear_test(layer_test, layer_ref, algebra_map, component_shape):
    hyper_dimension = len(algebra_map)

    components = []
    for _ in range(hyper_dimension):
        component = np.random.rand(*component_shape).astype('f')
        components.append(component)

    inp = np.concatenate(components)
    test_out = layer_test(inp)

    w = layer_test.get_weights()[0]
    w_components = np.split(w, hyper_dimension, axis=-1)

    layer_ref(components[0])
    bias = layer_ref.get_weights()[1]

    ref_partial = [[] for _ in range(hyper_dimension)]
    for i in range(hyper_dimension):
        layer_ref.set_weights([w_components[i], bias])
        for j in range(hyper_dimension):
            inter_res = layer_ref(components[j])
            ref_partial[i].append(inter_res)

    ref_outputs = [0 for _ in range(hyper_dimension)]
    for i in range(hyper_dimension):
        for j in range(hyper_dimension):
            which, coeff = algebra_map[i][j]
            ref_outputs[which] = ref_outputs[which] + ref_partial[i][j] * coeff

    ref_out = np.concatenate(ref_outputs)

#     print(test_out)
#     print(ref_out)

    assert_small_float_difference(test_out, ref_out, 0.0001)


# @pytest.mark.parametrize("layer_test, layer_ref", [
#     (generic_layers.Conv2D, tf.keras.layers.Conv2D),
#     # (generic_layers.DepthwiseConv2D, tf.keras.layers.DepthwiseConv2D),
#     # (generic_layers.Dense, tf.keras.layers.Dense),
# ])

@pytest.mark.parametrize('uptype', [0, 1, 2])
class TestGenericLinear:

    standard_params = [1, 2, 5, 16]

    @pytest.mark.parametrize('units', standard_params)
    @pytest.mark.parametrize('batch_size', standard_params)
    @pytest.mark.parametrize('channels', standard_params)
    def test_Dense(self, units, batch_size, channels, uptype):
        kwargs = {
            'units' : units
        }
        layer_test = generic_layers.Dense(*uptypes[uptype], **kwargs)
        layer_ref = tf.keras.layers.Dense(**kwargs)
        component_shape = (batch_size, channels)
        generic_linear_test(layer_test, layer_ref, algebra_maps[uptype], component_shape)