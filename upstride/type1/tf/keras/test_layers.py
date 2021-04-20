import pytest
import numpy as np
import tensorflow as tf
from upstride import generic_layers
from upstride.uptypes_utilities import UPTYPE1
from upstride.tests.utility import assert_small_float_difference, random_integer_tensor


standard_params = [1, 2, 5]

@pytest.mark.parametrize('units', standard_params)
@pytest.mark.parametrize('batch_size', standard_params)
@pytest.mark.parametrize('channels', standard_params)
def test_complex_Dense(units, batch_size, channels):
    layer_test = generic_layers.Dense(UPTYPE1, units=units, kernel_initializer=random_integer_tensor)

    shape = (batch_size, channels)
    real = random_integer_tensor(shape)
    imag = random_integer_tensor(shape)
    inp = tf.concat([real, imag], axis=0)
    test_out = layer_test(inp)

    comp_inp = real.numpy() + imag.numpy() * 1j
    w = layer_test.get_weights()[0]
    real_w, imag_w = [w[..., i::2] for i in range(2)]
    comp_w = real_w + imag_w * 1j
    comp_out = np.sum(np.array([comp_inp * comp_w[:, i] for i in range(units)]), axis=2)
    comp_out = np.transpose(comp_out)
    comp_out = np.concatenate((comp_out.real, comp_out.imag))

    assert_small_float_difference(test_out, comp_out, 0.0001)