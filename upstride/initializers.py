"""
This file implement the initialization function for a various type of operation and algebra

Ideas come from papers : 
  1- Deep Quaternion Networks https://arxiv.org/pdf/1712.04604.pdf (section E page 3-5)
  2- Deep Complex Networks https://arxiv.org/pdf/1705.09792.pdf (section 3.6 page 6)
  3- Quaternion recurrent neural network https://arxiv.org/pdf/1806.04418.pdf


# for 2, the trick is to say that 
Var(W) = Var(|W|) + (E[|W|])**2

and because it follow  Rayleigh distribution, 

E[|W|] = sigma * sqrt(pi/2)
Var(|W|) = (4-pi) * sigma**2 / 2

with sigma the Rayleigh distribution’s single parameter

for Glorot distribution, Var(W) = 2/(n_in + n_out) (n_in and n_out are the number of input and output units respectively)
solving the equation gives sigma = 1/sqrt(n_in + n_out)

for He distribution, Var(W) = 2/n_in, solving the equation gives sigma = 1/sqrt(n_in)

* magnitude of W is initialized using Rayleigh distribution with the good sigma
* phase came from random uniform distribution between -pi and pi

"""

import math
import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow.keras.initializers import Initializer


class InitializersFactory():
  def __init__(self):
    self.init_type0 = {
        'glorot_ind': IndependentFilter(),
        'he_ind': IndependentFilter(criterion='he'),
        'depthwise_glorot_ind': IndependentFilter(depthwise=True),
        'depthwise_he_ind': IndependentFilter(criterion='he', depthwise=True),
    }
    self.init_type1 = {
        'glorot_ind': IndependentFilter(complex=True),
        'he_ind': IndependentFilter(criterion='he', complex=True),
        'depthwise_glorot_ind': IndependentFilter(depthwise=True, complex=True),
        'depthwise_he_ind': IndependentFilter(criterion='he', depthwise=True, complex=True),
        'glorot': CInitializer(),
        'he': CInitializer(criterion='he'),
        'depthwise_glorot': CInitializer(depthwise=True),
        'depthwise_he': CInitializer(criterion='he', depthwise=True)
    }
    self.init_type2 = {
        'glorot': HInitializer(),
        'he': HInitializer(criterion='he'),
        'depthwise_glorot': HInitializer(depthwise=True),
        'depthwise_he': HInitializer(criterion='he', depthwise=True)
    }
    self.init_types = {0: self.init_type0, 1: self.init_type1, 2: self.init_type2}

  def is_custom_init(self, name):
    for k in list(self.init_types.keys()):
      if name in list(self.init_types[k]):
        return True
    return False

  def get_initializer(self, name, upstride_type):
    if name in list(self.init_types[upstride_type].keys()):
      return self.init_types[upstride_type][name]
    else:
      raise ValueError(f'Custom initializers {name} not supported for upstride_type = {upstride_type}')


# functions that return the variance of a specific criterion
# TODO : all framework doesn't seems to agree on this formulation, mainly the "2." is sometime "1."... We should benchmark both
# Here the convention used is the same than in the paper DCN and DQN
criterion_to_var = {
    'glorot': lambda n_in, n_out: 1. / (n_in + n_out),
    'he': lambda n_in, n_out: 1. / (n_in),
}


def get_input_output_unit(depthwise, shape):
  """ compute the n_in and n_out parameters used fot glorot or he criterion.
  These variables are fan_in and fan_out in Keras

  Note that here again, all frameworks implement different things. Here we try to stick as close as possible to the 
  intuition of the paper
  """
  if not depthwise:
    n_in, n_out = np.prod(shape[:-1]), shape[-1] * np.prod(shape[:-2])
  else:
    n_in, n_out = np.prod(shape[:-2]), shape[-1] * np.prod(shape[:-2])
  return n_in, n_out

# Please note that the above function is for python engine only. For cpp engine,
# the shape of the kernel is different so this function should change. It probably is the only
# thing in this file that needs to be adapted for Phoenix
#
# This is the version compatible with the cpp engine, (beware, not tested yet, use it at your own risk)
# def get_input_output_unit(depthwise, shape):
#   """
#   Dense layers have kernel shape (N_dim, C_in, C_out)
#   Conv layers have kernel shape (N_dim, O, I, [dim_list]) for instance Conv2D : (N_dim, O, I, H, W)
#   Depthwise conv layers have kernel shape (N_dim, depth_mul * I, 1, [dim_list]) for instance Conv2D : (N_dim, depth_mul * I, 1, H, W)
#   """
#   if len(shape) == 3:
#     # Then dense layer
#     n_in, n_out = shape[1], shape[2]
#   elif not depthwise:
#     # Then conv
#     n_in, n_out = np.prod(shape[2:]), shape[1]
#   else:
#     # then depthwise conv
#     n_in, n_out = np.prod(shape[2:]), shape[1]
#   return n_in, n_out


class IndependentFilter(Initializer):
  def __init__(self, criterion='glorot', depthwise=False, complex=False):
    """ Idea for the DCN paper
    This initialization constructs real-values or complex-values kernel with vectors that are independent as much as possible from each other while
    respecting either the He or the Glorot criterion. 

    Please note that this initialization is hard to implement in quaternion. If we measure any performance improvement for R or C, then
    we will implement it for H

    Note: for H we will need to implement "np.linalg.svd" ourselves

    """
    assert criterion in ['glorot', 'he'], f"Invalid criterion {criterion}"
    self.criterion = criterion
    self.depthwise = depthwise
    self.complex = complex

  def scale_filters(self, desired_var: float, independent_filters, shape):
    """scale the independent filters to get the desired variance

    Args:
        desired_var (float): wanted variance 
        independent_filters (numpy array): real or complex matrice with variance 1
    """
    # could be a lambda function, but I believe this formulation is clearer. If someone wants to change it feel free
    def scale(filters):
      return filters * np.sqrt(desired_var / np.var(filters))

    if self.complex:
      real = np.reshape(independent_filters.real, shape)
      img = np.reshape(independent_filters.imag, shape)
      return [scale(real), scale(img)]
    else:
      return [scale(independent_filters)]

  def reshape_weights(self, shape, scaled_filters):
    """ Operation to transpose and reshape the weight
    When calling this function, scaled_filters is a numpy array of shape (num_rows, num_cols)
    Need to transform it to an array of shape (num_cols, num_rows) and then resize it to array of kernel shape
    Note that for shape == 2, the resize operation is identity

    Args:
        shape (np.ndarray): shape of the kernel
        scaled_filters (np.array): filters to reshape

    Returns:
        reshaped filters
    """
    weights = np.transpose(scaled_filters, (1, 2, 0))
    weights = np.reshape(weights, shape)
    return weights

  def __call__(self, shape, dtype=None):
    """function called when initialing the kernel of a keras layer

    Args:
        shape (List[int]): Shape of the kernel tensor in initialized
          In case of a dense layer X -> Y, it will be [X, Y] (size of input, size of output)
          For convNd layer, shape is a list of N+2 ints, (kernel_1, kernel_2, ..., kernel_N, input, output)
          This shape is the same for both channel first and channel last
          For depthwise 2D convolution, shape = (kernel_x, kernel_y, n_channels, 1)
        dtype (type, optional): data type of the tensor
    """
    shape = list(shape)
    if self.complex:
      shape[-1] = int(shape[-1] / 2)  # divide per 2 to get shape in term of complex number
    if len(shape) == 2:  # then dense layer
      num_rows, num_cols = shape[0], shape[1]
    else:  # then Conv{1/2/3}D
      num_rows = shape[-1] * shape[-2]
      num_cols = np.prod(shape[:-2])  # product of all components of the kernel
    # generate the (semi-)unitary matrix
    if not self.complex:
      x = np.random.uniform(size=(num_rows, num_cols))
      u, _, v = np.linalg.svd(x)
      independent_filters = np.dot(u, np.dot(np.eye(num_rows, num_cols), v.T))
    else:
      rng = RandomState(seed=1337)
      x = rng.uniform(size=(num_rows, num_cols)) + 1j * rng.uniform(size=(num_rows, num_cols))
      u, _, v = np.linalg.svd(x)
      independent_filters = np.dot(u, np.dot(np.eye(num_rows, num_cols), np.conjugate(v).T))

    # n_in, n_out, the number of input and output units respectively defined in Glorot or He formula
    n_in, n_out = get_input_output_unit(self.depthwise, shape)
    desired_var = criterion_to_var[self.criterion](n_in, n_out)

    scaled_filters = self.scale_filters(desired_var, independent_filters, shape=(num_rows, ) + tuple(shape[:-2]))
    # At this step, scaled_filters is a list of tensor than contains one element if Real and 2 if complex

    # now need to reshape the kernel. If using complex then concatenate the real and imaginary components along the last axis
    weights = self.reshape_weights(shape, scaled_filters[0])
    if len(scaled_filters) == 2:
      weights = np.concatenate([weights, self.reshape_weights(shape, scaled_filters[1])], axis=-1)
    return weights


class CInitializer(Initializer):
  def __init__(self, criterion='glorot', depthwise=False):
    assert criterion in ['glorot', 'he'], f"Invalid criterion {criterion}"
    self.criterion = criterion
    self.depthwise = depthwise

  def __call__(self, shape, dtype=None):
    """function called then initialing the kernel of a keras layer

    Idea came from paper Deep Complex Networks https://arxiv.org/pdf/1705.09792.pdf (section 3.6 page 6)

    In short, the idea is to have a nice formulation of the variance by saying that 
    Var(W) = Var(|W|) + (E[|W|])**2

    and because it follow  Rayleigh distribution, 

    E[|W|] = sigma * sqrt(pi/2)
    Var(|W|) = (4-pi) * sigma**2 / 2

    with sigma the Rayleigh distribution’s single parameter

    for Glorot distribution, Var(W) = 2/(n_in + n_out) (n_in and n_out are the number of input and output units respectively)
    solving the equation gives sigma = 1/sqrt(n_in + n_out)

    for He distribution, Var(W) = 2/n_in, solving the equation gives sigma = 1/sqrt(n_in)

    * magnitude of W is initialized using Rayleigh distribution with the good sigma
    * phase is sample from an uniform distribution between -pi and pi

    Args:
        shape (List[int]): Shape of the kernel tensor in initialized
          In case of a dense layer X -> Y, it will be [X, Y] (size of input, size of output)
          For convNd layer, shape is a list of N+2 ints, (kernel_1, kernel_2, ..., kernel_N, input, output)
          This shape is the same for both channel first and channel last
          For depthwise 2D convolution, shape = (kernel_x, kernel_y, n_channels, 1)
        dtype (type, optional): data type of the tensor
    """
    shape = list(shape)
    shape[-1] = int(shape[-1] / 2)  # because complex, we divide by 2 here and concatenate along the last axis at the end
    n_in, n_out = get_input_output_unit(self.depthwise, shape)
    desired_var = criterion_to_var[self.criterion](n_in, n_out)
    sigma = np.sqrt(desired_var/2)
    magnitude = np.random.rayleigh(scale=sigma, size=shape)
    phase = np.random.uniform(low=-np.pi, high=np.pi, size=shape)
    # Complex ops are a bit special : return concatenation of real and img along last axis
    return np.concatenate([magnitude * np.cos(phase), magnitude * np.sin(phase)], axis=-1)


class HInitializer(Initializer):
  def __init__(self, criterion='glorot', depthwise=False):
    assert criterion in ['glorot', 'he'], f"Invalid criterion {criterion}"
    self.criterion = criterion
    self.depthwise = depthwise

  def __call__(self, shape, dtype=None):
    """
    see DQN paper for details regarding the math.

    Implementation is very similar to CInitializer except that the rayleigh distribution become a Chi4
    """
    shape = list(shape)
    shape[-1] = int(shape[-1] / 4)  # 4 because quaternion
    n_in, n_out = get_input_output_unit(self.depthwise, shape)
    desired_var = criterion_to_var[self.criterion](n_in, n_out)
    sigma = math.sqrt(desired_var/4)

    # Instead of implemented a Chi4 distribution, we can get 4 number from normal distribution, and get the norm
    # of the vector of these 4 components. This is maybe a bit more computational expensive but simpler to implement.
    # and as this function is called during the graph creation and not execution, we don't really care about speed ^^
    # (well, of course we care, but not a lot)
    r = np.random.normal(0., sigma, shape)
    i = np.random.normal(0., sigma, shape)
    j = np.random.normal(0., sigma, shape)
    k = np.random.normal(0., sigma, shape)
    magnitude = np.sqrt(r**2 + i**2 + j**2 + k**2)
    phase = np.random.uniform(low=-np.pi, high=np.pi, size=shape)

    # Create unit vector
    i = np.random.uniform(0., 1., shape)
    j = np.random.uniform(0., 1., shape)
    k = np.random.uniform(0., 1., shape)
    mag = np.sqrt(i**2+j**2+k**2)
    u_i = i / mag
    u_j = j / mag
    u_k = k / mag

    outputs = [
        magnitude * np.cos(phase),
        magnitude * u_i*np.sin(phase),
        magnitude * u_j*np.sin(phase),
        magnitude * u_k*np.sin(phase)
    ]
    return np.concatenate(outputs, axis=-1)
