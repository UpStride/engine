import tensorflow as tf
from upstride.generic_layers import unit_multiplier


class UpstrideDatatype:
  """
  Args:
      uptype_id:        an integer to serve as the Upstride datatype identifier. Kept for backward
                        compatibility. Preferrably try choose a value not already in use by other
                        datatypes.
      geometrical_def:  a tuple triplet containing the number of blades that square to 1, -1 and 0,
                        respectively, excluding the first blade (which is the scalar).
      blade_indexes:    a tuple containing strings that represent the multivector, in which '12'
                        represents e_{12}. The first element is expected to be an empty string ('')
                        representing the scalar. The other elements are expected to be consistent
                        among them (e.g. not have a bivector '13' if there is no vector '3') and
                        each string is expected to be written in an ascending fashion (e.g. '13'
                        (and NOT '31')).
  """
  def __init__(self, uptype_id: int, geometrical_def: tuple, blade_indexes: tuple):
    self.uptype_id = uptype_id
    self.geometrical_def = geometrical_def
    self.blade_indexes = blade_indexes

  @property
  def multivector_length(self) -> int:
    return len(self.blade_indexes)


UPTYPE0 = UpstrideDatatype(0, (0, 0, 0), ('',))
UPTYPE1 = UpstrideDatatype(1, (2, 0, 0), ('', '12'))
UPTYPE2 = UpstrideDatatype(2, (3, 0, 0), ('', '12', '23', '13'))
UPTYPE3 = UpstrideDatatype(3, (3, 0, 0), ('', '1', '2', '3', '12', '13', '23', '123'))


def prepare_inputs(uptype, inputs, channels_axis=1, **kwargs):
  # TODO consider implementing uptype.interlace so that inputs.shape is (BS*N, I, ...) instead of
  # inputs.shape (N*BS, I, ...)
  inputs = tf.reshape(inputs, [uptype.multivector_length, -1, *inputs.shape[1:]]) # shape (N, BS, I, ...)
  # Given that in a grouped convolution with g groups the input is splitted in g chunks along the
  # channels dimension (cf. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D),
  # then a special attention is required so that the multivector components do NOT get splitted
  # into different convolutions. The solution is to return a tensor of shape (BS, I*N, ...). As it
  # only matters for grouped convolution, the cheaper-to-compute tensor of shape (BS, N*I, ...) is
  # preferred whenever possible. This propagates to prepare_hyper_weight(), bias addition and
  # prepare_output().
  if channels_axis == 1: # inputs.shape (N, BS, I, ...)
    if kwargs.get('groups', 1) > 1:
      rest = list(range(3, tf.rank(inputs)))
      inputs = tf.transpose(inputs, perm=[1, 2, 0, *rest]) # shape (BS, I, N, ...)
      inputs = tf.reshape(inputs, [inputs.shape[0], -1, *inputs.shape[3:]]) # shape (BS, I*N, ...)
    else:
      rest = list(range(2, tf.rank(inputs)))
      inputs = tf.transpose(inputs, perm=[1, 0, *rest]) # shape (BS, N, I, ...)
      inputs = tf.reshape(inputs, [inputs.shape[0], -1, *inputs.shape[3:]]) # shape (BS, N*I, ...)
  else: # inputs.shape (N, BS, ..., I)
    rank = tf.rank(inputs)
    rest = list(range(2, rank - 1))
    if kwargs.get('groups', 1) > 1:
      inputs = tf.transpose(inputs, perm=[1, *rest, rank - 1, 0]) # shape (BS, ..., I, N)
      inputs = tf.reshape(inputs, [*inputs.shape[:-2], -1]) # shape (BS, ..., I*N)
    else:
      inputs = tf.transpose(inputs, perm=[1, *rest, 0, rank - 1]) # shape (BS, ..., N, I)
      inputs = tf.reshape(inputs, [*inputs.shape[:-2], -1]) # shape (BS, ..., N*I)
  return inputs


def prepare_output(uptype, output, channels_axis=1, **kwargs):
  if channels_axis == 1:
    if kwargs.get('groups', 1) > 1: # output.shape (BS, O*N, ...)
      output = tf.reshape(output, [output.shape[0], -1, uptype.multivector_length, *output.shape[2:]]) # shape (BS, O, N, ...)
      rest = list(range(3, tf.rank(output)))
      output = tf.transpose(output, perm=[2, 0, 1, *rest]) # shape (N, BS, O, ...)
    else: # output.shape (BS, N*O, ...)
      output = tf.reshape(output, [output.shape[0], uptype.multivector_length, -1, *output.shape[2:]]) # shape (BS, N, O, ...)
      rest = list(range(2, tf.rank(output)))
      output = tf.transpose(output, perm=[1, 0, *rest]) # shape (N, BS, O, ...)
    output = tf.reshape(output, [-1, *output.shape[2:]]) # shape (N*BS, O, ...)
  else:
    if kwargs.get('groups', 1) > 1: # output.shape (BS, ..., O*N)
      output = tf.reshape(output, [*output.shape[:-1], -1, uptype.multivector_length]) # shape (BS, ..., O, N)
      rank = tf.rank(output)
      rest = list(range(1, rank - 1))
      output = tf.transpose(output, perm=[rank - 1, 0, *rest]) # shape (N, BS, ..., O)
    else: # output.shape (BS, ..., N*O)
      output = tf.reshape(output, [*output.shape[:-1], uptype.multivector_length, -1]) # shape (BS, ..., N, O)
      rank = tf.rank(output)
      rest = list(range(1, rank - 2))
      output = tf.transpose(output, perm=[rank - 2, 0, *rest, rank - 1]) # shape (N, BS, ..., O)
    output = tf.reshape(output, [-1, *output.shape[2:]]) # shape (N*BS, ..., O)
  return output


def prepare_hyper_weight(uptype, weight, **kwargs):
  if uptype == UPTYPE2:
    kernels_4_r = tf.concat([weight[0], -weight[1], -weight[2], -weight[3]], axis=2)
    kernels_4_i = tf.concat([weight[1],  weight[0],  weight[3], -weight[2]], axis=2)
    kernels_4_j = tf.concat([weight[2], -weight[3],  weight[0],  weight[1]], axis=2)
    kernels_4_k = tf.concat([weight[3],  weight[2], -weight[1],  weight[0]], axis=2)
    hyper_weight = tf.concat([kernels_4_r, kernels_4_i, kernels_4_j, kernels_4_k], axis=3)
  elif uptype == UPTYPE1:
    kernels_2_r = tf.concat([weight[0], -weight[1]], axis=2)
    kernels_2_i = tf.concat([weight[1],  weight[0]], axis=2)
    hyper_weight = tf.concat([kernels_2_r, kernels_2_i], axis=3)
  else:
    # If a the type in use doesn't have a hard-coded hyper_weight, then
    # compute a single element A[i][j] for the matrix A that embeds the Hamilton product
    w_shape = weight.shape
    hyper_weight_list = []
    for i in range(uptype.multivector_length):
      hyper_weight_row = []
      for j in range(uptype.multivector_length):
        k, sign_i_j = unit_multiplier(uptype, i, j)
        _, sign_j_j = unit_multiplier(uptype, j, j)
        if sign_j_j == 0:
          raise ZeroDivisionError()
        hyper_weight_row.append(weight[k] * sign_i_j * sign_j_j)  # Given that sign_j_j is 1, 0 or -1,
        # a multiplication sign was preferred over the division. When it's 0, an error is raised.
      hyper_weight_list.append(tf.concat(hyper_weight_row, axis=2))
    hyper_weight = tf.concat(hyper_weight_list, axis=3)
  # hyper_weight.shape (..., N*I, N*O)
  if kwargs.get('groups', 1) > 1:
    shape = hyper_weight.shape
    updim = uptype.multivector_length
    intermediate_shape = [*shape[:-2], updim, shape[-2]//updim, updim, shape[-1]//updim]
    rank = tf.rank(hyper_weight) - 2
    rest = list(range(0, rank))
    hyper_weight = tf.reshape(hyper_weight, intermediate_shape) # shape (..., N, I, N, O)
    hyper_weight = tf.transpose(hyper_weight, perm=[*rest, rank + 1, rank, rank + 3, rank + 2]) # shape (..., I, N, O, N)
    hyper_weight = tf.reshape(hyper_weight, shape) # shape (..., I*N, O*N)
  return hyper_weight