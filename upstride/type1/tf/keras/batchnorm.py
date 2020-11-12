"""
Complex implementation of batch normalization from: https://arxiv.org/abs/1705.09792
Original implementation: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

this version perform the same operations as the deep complex network version, but has been rewritten to be cleaner
"""

from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
import numpy as np

def sqrt_init(shape, dtype=None):
  value = (1 / tf.sqrt(2.0)) * tf.ones(shape)
  return value

def complex_standardization(input_centred, v: Dict, axis=-1):
  """
  compute complex standardization.

    Args:
      input_centred (List): list of 2 tensors
      v (Dict): variance. Labels are 'rr', 'ii', 'ri'
      axis (int, optional): Defaults to -1.

    Returns:
      List: list of 2 tensors
  """

  # We require the covariance matrix's inverse square root. That first requires
  # square rooting, followed by inversion (I do this in that order because during
  # the computation of square root we compute the determinant we'll need for
  # inversion as well).

  # tau = v['rr'] + v['ii'] = Trace. Guaranteed >= 0 because SPD
  tau = v['rr'] + v['ii']
  # delta = (v['rr'] * v['ii']) - (v['ri'] ** 2) = Determinant. Guaranteed >= 0 because SPD
  delta = (v['rr'] * v['ii']) - (v['ri'] ** 2)

  s = tf.sqrt(delta) # Determinant of square root matrix
  t = tf.sqrt(tau + 2.0 * s)

  # The square root matrix could now be explicitly formed as
  #       [ v['rr']+s v['ri']   ]
  # (1/t) [ Vir   v['ii']+s ]
  # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
  # but we don't need to do this immediately since we can also simultaneously
  # invert. We can do this because we've already computed the determinant of
  # the square root matrix, and can thus invert it using the analytical
  # solution for 2x2 matrices
  #      [ A B ]             [  D  -B ]
  # inv( [ C D ] ) = (1/det) [ -C   A ]
  # http://mathworld.wolfram.com/MatrixInverse.html
  # Thus giving us
  #           [  v['ii']+s  -v['ri']   ]
  # (1/s)(1/t)[ -Vir     v['rr']+s ]
  # So we proceed as follows:

  inverse_st = 1.0 / (s * t)
  w = {}
  w['rr'] = (v['ii'] + s) * inverse_st
  w['ii'] = (v['rr'] + s) * inverse_st
  w['ri'] = -v['ri'] * inverse_st

  # And we have computed the inverse square root matrix W = sqrt(V)!
  # Normalization. We multiply, x_normalized = W.x.

  # The returned result will be a complex standardized input
  # where the real and imaginary parts are obtained as follows:
  # x_real_normed = w['rr'] * x_real_centred + w['ri'] * x_imag_centred
  # x_imag_normed = w['ri'] * x_real_centred + w['ii'] * x_imag_centred

  output = [w['rr'] * input_centred[0] + w['ri'] * input_centred[1], 
        w['ri'] * input_centred[0] + w['ii'] * input_centred[1]]

  return output


def complex_bn(input_centred, v: Dict, beta, gamma: Dict, scale=True,
         center=True, layernorm=False, axis=-1):

  standardized_output = complex_standardization(input_centred, v, axis=axis)

  # Now we perform th scaling and Shifting of the normalized x using
  # the scaling parameter
  #           [  gamma['rr'] gamma['ri']  ]
  #   Gamma = [  gamma['ri'] gamma['ii']  ]
  # and the shifting parameter
  #    Beta = [beta_real beta_imag].T
  # where:
  # x_real_BN = gamma['rr'] * x_real_normed + gamma['ri'] * x_imag_normed + beta_real
  # x_imag_BN = gamma['ri'] * x_real_normed + gamma['ii'] * x_imag_normed + beta_imag
  
  broadcast_beta_shape = [1] * len(input_centred[0].shape)
  broadcast_beta_shape[axis] = input_centred[0].shape[axis]

  real_output = gamma['rr'] * standardized_output[0] + gamma['ri'] * standardized_output[1]
  real_output += tf.reshape(beta['r'], broadcast_beta_shape)
  imag_output = gamma['ri'] * standardized_output[0] + gamma['ii'] * standardized_output[1]
  imag_output += tf.reshape(beta['i'], broadcast_beta_shape)

  return [real_output, imag_output]






class BatchNormalizationC(Layer):
  """Complex version of the real domain 
  Batch normalization layer (Ioffe and Szegedy, 2014).
  Normalize the activations of the previous complex layer at each batch,
  i.e. applies a transformation that maintains the mean of a complex unit
  close to the null vector, the 2 by 2 covariance matrix of a complex unit close to identity
  and the 2 by 2 relation matrix, also called pseudo-covariance, close to the 
  null matrix.
  # Arguments
    axis: Integer, the axis that should be normalized
      (typically the features axis).
      For instance, after a `Conv2D` layer with
      `data_format="channels_first"`,
      set `axis=2` in `ComplexBatchNormalization`.
    momentum: Momentum for the moving statistics related to the real and
      imaginary parts.
    epsilon: Small float added to each of the variances related to the
      real and imaginary parts in order to avoid dividing by zero.
    center: If True, add offset of `beta` to complex normalized tensor.
      If False, `beta` is ignored.
      (beta is formed by real_beta and imag_beta)
    scale: If True, multiply by the `gamma` matrix.
      If False, `gamma` is not used.
    beta_initializer: Initializer for the real_beta and the imag_beta weight.
    gamma_diag_initializer: Initializer for the diagonal elements of the gamma matrix.
      which are the variances of the real part and the imaginary part.
    gamma_off_initializer: Initializer for the off-diagonal elements of the gamma matrix.
    moving_mean_initializer: Initializer for the moving means.
    moving_variance_initializer: Initializer for the moving variances.
    moving_covariance_initializer: Initializer for the moving covariance of
      the real and imaginary parts.
    beta_regularizer: Optional regularizer for the beta weights.
    gamma_regularizer: Optional regularizer for the gamma weights.
    beta_constraint: Optional constraint for the beta weights.
    gamma_constraint: Optional constraint for the gamma weights.
  # Input shape
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  # Output shape
    Same shape as input.
  # References
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  def __init__(self, axis=-1, momentum=0.9, epsilon=1e-4, center=True, scale=True, beta_initializer='zeros',
         gamma_diag_initializer='sqrt_init',
         gamma_off_initializer='zeros',
         moving_mean_initializer='zeros',
         moving_variance_initializer='sqrt_init',
         moving_covariance_initializer='zeros',
         beta_regularizer=None,
         gamma_diag_regularizer=None,
         gamma_off_regularizer=None,
         beta_constraint=None,
         gamma_diag_constraint=None,
         gamma_off_constraint=None,
         **kwargs):
    super(BatchNormalizationC, self).__init__(**kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale

    # gamma parameter, diagonale (trainable)
    if gamma_diag_initializer != 'sqrt_init':
      self.gamma_diag_initializer = tf.keras.initializers.get(gamma_diag_initializer)
    else:
      self.gamma_diag_initializer = sqrt_init
    self.gamma_diag_regularizer = tf.keras.regularizers.get(gamma_diag_regularizer)
    self.gamma_diag_constraint = tf.keras.constraints.get(gamma_diag_constraint)

    # gamma parameter, outside of diagonale (trainable)
    self.gamma_off_initializer = tf.keras.initializers.get(gamma_off_initializer)
    self.gamma_off_regularizer = tf.keras.regularizers.get(gamma_off_regularizer)
    self.gamma_off_constraint = tf.keras.constraints.get(gamma_off_constraint)

    # beta parameter (trainable)
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)

    # moving_V parameter (not trainable)
    if moving_variance_initializer != 'sqrt_init':
      self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
    else:
      self.moving_variance_initializer = sqrt_init

    # moving covariance
    self.moving_covariance_initializer = tf.keras.initializers.get(moving_covariance_initializer)

    # moving_mean (not trainable)
    self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)

  def build(self, input_shape):
    ndim = len(input_shape[0])
    # update self.axis
    if self.axis < 0:
      self.axis = ndim + self.axis  # usually 3
    param_shape = input_shape[0][self.axis]  # 5 for unit-tests
    if param_shape is None:
      raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
               f'but the layer received an input with shape {input_shape}.')

    self.gamma = {}
    self.moving_V = {}
    for postfix in ['rr','ri','ii']:
      self.gamma[postfix] = self.add_weight(shape=param_shape,
                          name=f'gamma_{postfix}',
                          initializer=self.gamma_diag_initializer,
                          regularizer=self.gamma_diag_regularizer,
                          constraint=self.gamma_diag_constraint)
      self.moving_V[postfix] = self.add_weight(shape=param_shape,
                           initializer=self.moving_variance_initializer,
                           name=f'moving_V{postfix}',
                           trainable=False)

    self.beta = {}
    self.moving_mean = []
    for postfix in ['r','i']:
      self.beta[postfix] = self.add_weight(shape=(input_shape[0][self.axis],),
                         name=f'beta{[postfix]}',
                         initializer=self.beta_initializer,
                         regularizer=self.beta_regularizer,
                         constraint=self.beta_constraint)
      self.moving_mean.append(self.add_weight(shape=(input_shape[0][self.axis],),
                          initializer=self.moving_mean_initializer,
                          name=f'moving_mean{[postfix]}',
                          trainable=False))

    self.built = True

  def _get_training_value(self, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    if isinstance(training, int):
      training = bool(training)
    if not self.trainable:
      # When the layer is not trainable, it overrides the value passed from
      # model.
      training = False
    return training

  def call(self, inputs, training=None):
    training = self._get_training_value(training)

    # inputs is an array of 2 tensors
    input_shape = inputs[0].shape  # typically [BS, H, W, C]. For unittest (1,2,3,5)
    ndims = len(input_shape)  # typically 4
    reduction_axes = [i for i in range(ndims) if i != self.axis]  # [0, 1, 2]

    # substract by mean
    mu = []
    broadcast_mu = []
    broadcast_mu_shape = [1] * ndims
    broadcast_mu_shape[self.axis] = input_shape[self.axis]  # unittest [1, 1, 1, 5]
    for i in range(2):
      mu.append(tf.math.reduce_mean(inputs[i], axis=reduction_axes))  # compute mean for all blades. Unittest gives tf.Tensor([1 3 4 5 6], shape=(5,), dtype=int32)
      broadcast_mu.append(tf.reshape(mu[i], broadcast_mu_shape))
    input_centred = [inputs[i] - broadcast_mu[i] for i in range(2)]

    v = {}
    if self.scale:
      v['rr'] = tf.reduce_mean(input_centred[0] * input_centred[0], axis=reduction_axes) \
          + self.epsilon
      v['ii'] = tf.reduce_mean(input_centred[1] * input_centred[1], axis=reduction_axes) \
          + self.epsilon
      v['ri'] = tf.reduce_mean(input_centred[0] * input_centred[1], axis=reduction_axes)
    elif self.center:
      v['rr'] = None
      v['ii'] = None
      v['ri'] = None
    else:
      raise ValueError('Error. Both scale and center in batchnorm are set to False.')

    input_bn = complex_bn(
      input_centred, v,
      self.beta, 
      self.gamma,
      axis=self.axis
    )
    training_value = tf_utils.constant_value(training)
    if training_value == False:  # not the same as "not training_value" because of possible none
      return input_bn

    update_list = []
    update_list.append(tf.keras.backend.moving_average_update(self.moving_mean[0], mu[0], self.momentum))
    update_list.append(tf.keras.backend.moving_average_update(self.moving_mean[1], mu[1], self.momentum))
    update_list.append(tf.keras.backend.moving_average_update(self.moving_V['rr'], v['rr'], self.momentum))
    update_list.append(tf.keras.backend.moving_average_update(self.moving_V['ii'], v['rr'], self.momentum))
    update_list.append(tf.keras.backend.moving_average_update(self.moving_V['ri'], v['rr'], self.momentum))
    self.add_update(update_list, inputs)

    def normalize_inference():
      if self.center:
        inference_centred = [inputs[0] - tf.reshape(self.moving_mean[0], broadcast_mu_shape),
                   inputs[1] - tf.reshape(self.moving_mean[1], broadcast_mu_shape)]
      else:
        inference_centred = inputs
      return complex_bn(
        inference_centred, self.moving_V, 
        self.beta, 
        self.gamma, 
        axis=self.axis
      )

    # Pick the normalized form corresponding to the training phase.
    return tf.keras.backend.in_train_phase(input_bn, normalize_inference, training=training)

  def get_config(self):
    config = {
      'axis': self.axis,
      'momentum': self.momentum,
      'epsilon': self.epsilon,
      'center': self.center,
      'scale': self.scale,
      'beta_initializer': initializers.serialize(self.beta_initializer),
      'gamma_diag_initializer': initializers.serialize(self.gamma_diag_initializer) if self.gamma_diag_initializer != sqrt_init else 'sqrt_init',
      'gamma_off_initializer': initializers.serialize(self.gamma_off_initializer),
      'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
      'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer) if self.moving_variance_initializer != sqrt_init else 'sqrt_init',
      'moving_covariance_initializer': initializers.serialize(self.moving_covariance_initializer),
      'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
      'gamma_diag_regularizer': tf.keras.regularizers.serialize(self.gamma_diag_regularizer),
      'gamma_off_regularizer': tf.keras.regularizers.serialize(self.gamma_off_regularizer),
      'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
      'gamma_diag_constraint': tf.keras.constraints.serialize(self.gamma_diag_constraint),
      'gamma_off_constraint': tf.keras.constraints.serialize(self.gamma_off_constraint),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))