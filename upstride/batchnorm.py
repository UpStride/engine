"""
This file implements special batch normalization for complex and quaterion cases
"""
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


def get_sqrt_init(multivector_length):
  value = 1 / tf.sqrt(float(multivector_length))
  return lambda shape, dtype: tf.cast(value * tf.ones(shape), dtype)


class GenericBatchNormalization(tf.keras.layers.Layer):
  """ This class implement the generic architecture of a batch normalization.
  It need to be inherited to implements the specificities of same geometrical algebra

  The signature of the init function is the same that BatchNormalization of tensorflow 2.3/2.4
  """

  def __init__(self,
               axis=1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='sqrt_init',  # The default value of this parameter is different from TF
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               trainable=True,
               **kwargs):
    super().__init__(**kwargs)
    # this version of batch normalization implements a new security : if checked if axis is meaningful with Keras data_format.
    # If it is not, then print a warning
    if tf.keras.backend.image_data_format() == 'channels_first' and axis == -1:
      print('Warning, Batchnormalization is called on channels -1 when data format is "channels_first". Do you really want to to this ?')
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center  # if true then use beta
    self.scale = scale  # if true then multiply by gamma
    self.beta_initializer = beta_initializer
    self.gamma_initializer = gamma_initializer
    self.moving_mean_initializer = moving_mean_initializer
    self.moving_variance_initializer_str = moving_variance_initializer
    self.beta_regularizer_srt = beta_regularizer
    self.gamma_regularizer = gamma_regularizer
    self.beta_constraint_str = beta_constraint
    self.gamma_constraint = gamma_constraint

    self.dim_names = 'rijk'[:self.multivector_length]

    # gamma parameter, diagonal (trainable)
    self.gamma_diag_initializer = get_sqrt_init(self.multivector_length) if gamma_initializer == 'sqrt_init' else tf.keras.initializers.get(gamma_initializer)
    self.gamma_diag_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self.gamma_diag_constraint = tf.keras.constraints.get(gamma_constraint)

    # gamma parameter, outside of diagonal (trainable)
    self.gamma_off_initializer = tf.keras.initializers.get('zeros')
    self.gamma_off_regularizer = tf.keras.regularizers.get(gamma_regularizer)
    self.gamma_off_constraint = tf.keras.constraints.get(gamma_constraint)

    # beta parameter (trainable)
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
    self.beta_constraint = tf.keras.constraints.get(beta_constraint)

    # moving_V parameter (not trainable)
    self.moving_variance_initializer = get_sqrt_init(
        self.multivector_length) if moving_variance_initializer == 'sqrt_init' else tf.keras.initializers.get(moving_variance_initializer)

    # moving covariance (not trainable)
    self.moving_covariance_initializer = tf.keras.initializers.get('zeros')

    # moving_mean (not trainable)
    self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)

  def build(self, input_shape):
    ndim = len(input_shape)  # usually 4
    # update self.axis
    if self.axis < 0:
      self.axis = ndim + self.axis  # usually 1 if channel first, 3 if channel last
    param_shape = input_shape[self.axis]
    if param_shape is None:
      raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
                       f'but the layer received an input with shape {input_shape}.')
    self.gamma = {}
    self.moving_V = {}
    for p1 in range(self.multivector_length):
      for p2 in range(p1, self.multivector_length):
        postfix = self.dim_names[p1]+self.dim_names[p2]
        if self.scale:
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
    for p1 in range(self.multivector_length):
      postfix = self.dim_names[p1]
      if self.center:
        self.beta[postfix] = self.add_weight(shape=(param_shape,),
                                             name=f'beta{[postfix]}',
                                             initializer=self.beta_initializer,
                                             regularizer=self.beta_regularizer,
                                             constraint=self.beta_constraint)
      self.moving_mean.append(self.add_weight(shape=(param_shape,),
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
      # When the layer is not trainable, it overrides the value passed from model.
      training = False
    return training

  def compute_mean_var(self, inputs):
    input_shape = inputs[0].shape  # typically [BS, H, W, C]
    ndims = len(input_shape)  # typically 4
    reduction_axes = [i for i in range(ndims) if i != self.axis]  # [0, 1, 2]

    # substract by mean
    mu = []
    broadcast_mu = []
    broadcast_mu_shape = [1] * ndims
    broadcast_mu_shape[self.axis] = input_shape[self.axis]
    for i in range(self.multivector_length):
      mu.append(tf.math.reduce_mean(inputs[i], axis=reduction_axes))  # compute mean for all blades.
      broadcast_mu.append(tf.reshape(mu[i], broadcast_mu_shape))
    input_centred = [inputs[i] - broadcast_mu[i] for i in range(self.multivector_length)]

    # compute covariance matrix
    v = {}
    for p1 in range(self.multivector_length):
      for p2 in range(p1, self.multivector_length):
        postfix = self.dim_names[p1]+self.dim_names[p2]
        v[postfix] = tf.reduce_mean(input_centred[p1] * input_centred[p2], axis=reduction_axes)
        if p1 == p2:
          v[postfix] += self.epsilon

    return input_centred, mu, v, broadcast_mu_shape

  def call(self, inputs, training=None):
    # inputs is an array of 'multivector_length' tensors
    inputs = tf.split(inputs, self.multivector_length, axis=0)
    training = self._get_training_value(training)
    input_centred, mu, v, broadcast_mu_shape = self.compute_mean_var(inputs)

    input_bn = self.bn(input_centred, v)
    training_value = tf_utils.constant_value(training)
    if training_value == False:  # not the same as "not training_value" because of possible none
      return tf.concat(input_bn, axis=0)

    update_list = []
    for i in range(self.multivector_length):
      update_list.append(tf.keras.backend.moving_average_update(self.moving_mean[i], mu[i], self.momentum))
    for p1 in range(self.multivector_length):
      for p2 in range(p1, self.multivector_length):
        postfix = self.dim_names[p1]+self.dim_names[p2]
        update_list.append(tf.keras.backend.moving_average_update(self.moving_V[postfix], v[postfix], self.momentum))
    self.add_update(update_list, inputs)

    def normalize_inference():
      inference_centred = [inputs[i] - tf.reshape(self.moving_mean[i], broadcast_mu_shape) for i in range(self.multivector_length)]
      return self.bn(inference_centred, v)

    # Pick the normalized form corresponding to the training phase.
    return tf.concat(tf.keras.backend.in_train_phase(input_bn, normalize_inference, training=training), axis=0)

  def bn(self, input_centred, v):
    w = self.compute_sqrt_inv(v)

    broadcast_beta_shape = [1] * len(input_centred[0].shape)
    num_channels = input_centred[0].shape[self.axis]
    broadcast_beta_shape[self.axis] = num_channels # unittest [1, 1, 1, 5]

    # Normalization. We multiply, x_normalized = W.x.
    # The returned result will be a quaternion/complex standardized input
    # For quaternions, r, i, j, and k parts are obtained as follows:
    # x_r_normed = Wrr * x_r_cent + Wri * x_i_cent + Wrj * x_j_cent + Wrk * x_k_cent
    # x_i_normed = Wri * x_r_cent + Wii * x_i_cent + Wij * x_j_cent + Wik * x_k_cent
    # x_j_normed = Wrj * x_r_cent + Wij * x_i_cent + Wjj * x_j_cent + Wjk * x_k_cent
    # x_k_normed = Wrk * x_r_cent + Wik * x_i_cent + Wjk * x_j_cent + Wkk * x_k_cent
    output = []
    for p1 in range(self.multivector_length):
      tmp_out = 0
      for p2 in range(self.multivector_length):
        i1 = min(p1, p2)
        i2 = max(p1, p2)
        tmp_out += tf.reshape(w[self.dim_names[i1]+self.dim_names[i2]], broadcast_beta_shape) * input_centred[p2]
      output.append(tmp_out)
    standardized_output = output

    # Now we perform the scaling and shifting of the normalized x using
    # the scaling parameter
    #           [  gamma_rr gamma_ri gamma_rj gamma_rk  ]
    #   Gamma = [  gamma_ri gamma_ii gamma_ij gamma_ik  ]
    #           [  gamma_rj gamma_ij gamma_jj gamma_jk  ]
    #           [  gamma_rk gamma_ik gamma_jk gamma_kk  ]
    # and the shifting parameter
    #    Beta = [beta_r beta_i beta_j beta_k].T
    # where:
    # x_r_BN = gamma_rr * x_r + gamma_ri * x_i + gamma_rj * x_j + gamma_rk * x_k + beta_r
    # x_i_BN = gamma_ri * x_r + gamma_ii * x_i + gamma_ij * x_j + gamma_ik * x_k + beta_i
    # x_j_BN = gamma_rj * x_r + gamma_ij * x_i + gamma_jj * x_j + gamma_jk * x_k + beta_j
    # x_k_BN = gamma_rk * x_r + gamma_ik * x_i + gamma_jk * x_j + gamma_kk * x_k + beta_k

    # first multiply by gamma
    output = []
    if self.scale:
      for p1 in range(self.multivector_length):
        tmp_out = 0
        for p2 in range(self.multivector_length):
          i1 = min(p1, p2)
          i2 = max(p1, p2)
          ratio = self.gamma[self.dim_names[i1]+self.dim_names[i2]]
          tmp_out += tf.reshape(ratio, broadcast_beta_shape) * standardized_output[p2]
        output.append(tmp_out)
    else:
      output = standardized_output
    # then add beta
    input = output
    output = []
    if self.center:
      for p1 in range(self.multivector_length):
        delta = self.beta[self.dim_names[p1]]
        delta = tf.reshape(delta, broadcast_beta_shape) 
        output.append(input + delta)
    else:
      output = input
        
    return output

  def get_config(self):
    config = {
        'axis': self.axis,
        'momentum': self.momentum,
        'epsilon': self.epsilon,
        'center': self.center,
        'scale': self.scale,
        'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
        'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer) if self.gamma_initializer != 'sqrt_init' else 'sqrt_init',
        'moving_mean_initializer': tf.keras.initializers.serialize(self.moving_mean_initializer),
        'moving_variance_initializer': tf.keras.initializers.serialize(self.moving_variance_initializer_str) if self.moving_variance_initializer_str != 'sqrt_init' else 'sqrt_init',
        'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer_srt),
        'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
        'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint_str),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchNormalizationH(GenericBatchNormalization):
  """
  quaternion implementation : https://github.com/gaudetcj/DeepQuaternionNetworks/blob/43b321e1701287ce9cf9af1eb16457bdd2c85175/quaternion_layers/bn.py
  tf implementation : https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
  paper : https://arxiv.org/pdf/1712.04604.pdf

  this version perform the same operations as the deep quaternion network version, but has been rewritten to be cleaner
  """

  def __init__(self, *argv, **kwargs):
    self.multivector_length = 4
    super().__init__(*argv, **kwargs)

  def compute_sqrt_inv(self, v):
    # Chokesky decomposition of 4x4 symmetric matrix
    # see paper https://arxiv.org/pdf/1712.04604.pdf appendix C for details
    w = {}
    w['rr'] = tf.sqrt(v['rr'])
    wrr_inverse = 1. / w['rr']
    w['ri'] = wrr_inverse * (v['ri'])
    w['rj'] = wrr_inverse * (v['rj'])
    w['rk'] = wrr_inverse * (v['rk'])
    w['ii'] = tf.sqrt((v['ii'] - (w['ri']*w['ri'])))
    wii_inverse = 1. / w['ii']
    w['ij'] = wii_inverse * (v['ij'] - (w['ri']*w['rj']))
    w['ik'] = wii_inverse * (v['ik'] - (w['ri']*w['rk']))
    w['jj'] = tf.sqrt((v['jj'] - (w['ij']*w['ij'] + w['rj']*w['rj'])))
    w['jk'] = (1.0 / w['jj']) * (v['jk'] - (w['ij']*w['ik'] + w['rj']*w['rk']))
    w['kk'] = tf.sqrt((v['kk'] - (w['jk']*w['jk'] + w['ik']*w['ik'] + w['rk']*w['rk'])))

    # # compute the opposite
    # # basic version is
    # o = {}
    # o['rr'] = 1 / w['rr']
    # o['ii'] = 1 / w['ii']
    # o['jj'] = 1 / w['jj']
    # o['kk'] = 1 / w['kk']
    # o['ri'] = -(w['ri'] * o['rr']) / w['ii']
    # o['rj'] = -(w['rj'] * o['rr'] + w['ij'] * o['ri']) / w['jj']
    # o['rk'] = -(w['rk'] * o['rr'] + w['ik'] * o['ri'] + w['jk'] * o['rj'])/w['kk']
    # o['ij'] = -(w['ij'] * o['ii']) / w['jj']
    # o['ik'] = -(w['ik'] * o['ii'] + w['jk'] * o['ij']) / w['kk']
    # o['jk'] = -(w['jk'] * o['jj']) / w['kk']

    # with less division it gives
    o = {}
    o['rr'] = 1 / w['rr']
    o['ii'] = 1 / w['ii']
    o['jj'] = 1 / w['jj']
    o['kk'] = 1 / w['kk']
    o['ri'] = -(w['ri'] * o['rr']) * o['ii']
    o['rj'] = -(w['rj'] * o['rr'] + w['ij'] * o['ri']) * o['jj']
    o['rk'] = -(w['rk'] * o['rr'] + w['ik'] * o['ri'] + w['jk'] * o['rj']) * o['kk']
    o['ij'] = -(w['ij'] * o['ii']) * o['jj']
    o['ik'] = -(w['ik'] * o['ii'] + w['jk'] * o['ij']) * o['kk']
    o['jk'] = -(w['jk'] * o['jj']) * o['kk']

    return o


class BatchNormalizationC(GenericBatchNormalization):
  """Complex version of the real domain 
  Complex implementation of batch normalization from: https://arxiv.org/abs/1705.09792
  Original implementation: https://github.com/ChihebTrabelsi/deep_complex_networks/blob/master/complexnn/bn.py

  this version perform the same operations as the deep complex network version, but has been rewritten to be cleaner

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

  def __init__(self, *argv, **kwargs):
    self.multivector_length = 2
    super().__init__(*argv, **kwargs)

  def compute_sqrt_inv(self, v):
    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = v['rr'] + v['ii'] = Trace. Guaranteed >= 0 because SPD
    tau = v['rr'] + v['ii']
    # delta = (v['rr'] * v['ii']) - (v['ri'] ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (v['rr'] * v['ii']) - (v['ri'] ** 2)

    s = tf.sqrt(delta)  # Determinant of square root matrix
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
    return w
