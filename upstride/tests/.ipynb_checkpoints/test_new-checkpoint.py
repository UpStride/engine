import pytest
import tensorflow as tf
import numpy as np
from upstride import generic_layers


# def test_dense_simplest():
#     l = generic_layers.Dense(1, ['', '12'], (2, 0, 0), units=1, kernel_initializer='ones')
#     img = np.ones((2, 1))
#     out = l(img)
#     ref = [[0], [2]]
#     assert np.all(out == ref)


def test_dense():
    l = generic_layers.Dense(1, ['', '12'], (2, 0, 0), units=1, kernel_initializer='ones')
    tensor_len = 2
    real = np.random.rand(tensor_len)
    imaginary = np.random.rand(tensor_len)
    img = np.concatenate((real, imaginary))
    img = np.expand_dims(img, axis=1)
    comp = real + imaginary * 1j
    out = l(img)


    print(img)
    print(comp)
    print(out)