import tensorflow as tf


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