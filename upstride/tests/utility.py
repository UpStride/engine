import numpy as np
import tensorflow as tf
from upstride.uptypes_utilities import UPTYPE0, UPTYPE1, UPTYPE2, UPTYPE3, UpstrideDatatype


UPTYPE2_alternative = UpstrideDatatype(2, (0, 2, 0), ['', '1', '2', '12'])

uptypes = {
    'up0' : UPTYPE0,
    'up1' : UPTYPE1,
    'up2' : UPTYPE2,
    'up3' : UPTYPE3,
    'up2_alt' : UPTYPE2_alternative,
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


def gpu_visible():
    """ Returns True if TF sees GPU
    """
    return tf.config.list_physical_devices('GPU') != []


def assert_small_float_difference(tensor1, tensor2, relative_error_threshold=0.001):
    """ Asserts float tensors differ by no more than threshold scaled by the values checked
    """
    abs_diff = tf.abs(tensor1 - tensor2)
    abs_max_tensors = tf.abs(tf.maximum(tensor1, tensor2))
    threshold = relative_error_threshold * (1 + abs_max_tensors)
    assert tf.reduce_all(abs_diff < threshold)


def random_float_tensor(shape, dtype=tf.float32):
    return tf.random.uniform(shape, dtype=dtype)


def random_integer_tensor(shape, dtype=tf.float32):
    return tf.cast(tf.random.uniform(shape, -4, +4, dtype=tf.int32), dtype)


def create_components(hyper_dimension, component_shape, random_tensor=random_integer_tensor):
    components = []
    for _ in range(hyper_dimension):
        component = random_tensor(component_shape)
        components.append(component)
    return components