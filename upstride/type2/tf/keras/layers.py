from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from .... import generic_layers
from .... generic_layers import *
from .convolutional import Conv2D
from .dense import Dense

generic_layers.upstride_type = 2
generic_layers.blade_indexes = ["", "12", "23", "13"]
generic_layers.geometrical_def = (3, 0, 0)

# If you wish to overwrite some layers, please implements them here


class TF2Upstride(Layer):
    """assume this function is called at the begining of the network. Put colors to imaginary parts and grayscale in real
    """

    def __init__(self, strategy=''):
        self.rgb_in_img = False
        self.gray_in_real_rgb_in_img = False
        if strategy == "rgbinimg":
            self.rgb_in_img = True
        elif strategy == 'grayinreal_rgbinimg':
            self.gray_in_real_rgb_in_img = True

    def __call__(self, x):
        if self.rgb_in_img:
            red = tf.expand_dims(x[:, :, :, 0], -1)
            green = tf.expand_dims(x[:, :, :, 1], -1)
            blue = tf.expand_dims(x[:, :, :, 2], -1)
            zeros = tf.zeros_like(red)
            return [zeros, red, green, blue]
        elif self.gray_in_real_rgb_in_img:
            red = tf.expand_dims(x[:, :, :, 0], -1)
            green = tf.expand_dims(x[:, :, :, 1], -1)
            blue = tf.expand_dims(x[:, :, :, 2], -1)
            grayscale = (red + green + blue)/3
            return [grayscale, red, green, blue]
        else:
            return [x]


def sqrt_init(shape, dtype=None):
    value = (1 / tf.sqrt(4.0)) * tf.ones(shape)
    return value


class BatchNormalizationUnfinised(Layer):
    """
    quaternion implementation : https://github.com/gaudetcj/DeepQuaternionNetworks/blob/43b321e1701287ce9cf9af1eb16457bdd2c85175/quaternion_layers/bn.py
    tf implementation : https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/normalization.py#L46
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
        """
        Args:
            axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a `Conv2D` layer with
                `data_format="channels_first"`, set `axis=2` in `QuaternionBatchNormalization`.
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center  # if true then use beta and gamma to add a bit of variance
        self.scale = scale

        self.gamma_off_initializer = initializers.get(gamma_off_initializer)
        self.moving_covariance_initializer = initializers.get(moving_covariance_initializer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)

        # gamma parameter (trainable)
        if gamma_diag_initializer != 'sqrt_init':
            self.gamma_diag_initializer = initializers.get(gamma_diag_initializer)
        else:
            self.gamma_diag_initializer = sqrt_init
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)

        # beta parameter (trainable)
        self.beta_initializer = initializers.get(beta_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)

        # moving_V parameter (not trainable)
        if moving_variance_initializer != 'sqrt_init':
            self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        else:
            self.moving_variance_initializer = sqrt_init

        # moving_mean (not trainable)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)

    def build(self, input_shape):
        ndim = len(input_shape)
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(f'Axis {self.axis} of input tensor should have a defined dimension '
                             f'but the layer received an input with shape {input_shape}.')
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        param_shape = (input_shape[self.axis] // 4,)

        self.gamma = {}
        self.moving_V = {}
        dim_names = "rijk"

        for p1 in range(4):
            for p2 in range(p1, 4):
                postfix = dim_names[pi]+dim_names[p2]
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
                else:
                    self.gamma[postfix] = None
                    self.moving_V[postfix] = None
        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],), name='beta', initializer=self.beta_initializer, regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape[self.axis],), initializer=self.moving_mean_initializer, name='moving_mean',
                                               trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = inputs.shape
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 4  # TODO or not ?
        mu = K.mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_r = centred_squared[:, :input_dim]
            centred_squared_i = centred_squared[:, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, input_dim*3:]
            centred_r = input_centred[:, :input_dim]
            centred_i = input_centred[:, input_dim:input_dim*2]
            centred_j = input_centred[:, input_dim*2:input_dim*3]
            centred_k = input_centred[:, input_dim*3:]
        elif ndim == 3:
            centred_squared_r = centred_squared[:, :, :input_dim]
            centred_squared_i = centred_squared[:, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, input_dim*3:]
            centred_r = input_centred[:, :, :input_dim]
            centred_i = input_centred[:, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, input_dim*3:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_r = centred_squared[:, :, :, :input_dim]
            centred_squared_i = centred_squared[:, :, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, :, input_dim*3:]
            centred_r = input_centred[:, :, :, :input_dim]
            centred_i = input_centred[:, :, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, :, input_dim*3:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_r = centred_squared[:, :, :, :, :input_dim]
            centred_squared_i = centred_squared[:, :, :, :, input_dim:input_dim*2]
            centred_squared_j = centred_squared[:, :, :, :, input_dim*2:input_dim*3]
            centred_squared_k = centred_squared[:, :, :, :, input_dim*3:]
            centred_r = input_centred[:, :, :, :, :input_dim]
            centred_i = input_centred[:, :, :, :, input_dim:input_dim*2]
            centred_j = input_centred[:, :, :, :, input_dim*2:input_dim*3]
            centred_k = input_centred[:, :, :, :, input_dim*3:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_r,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_i,
                axis=reduction_axes
            ) + self.epsilon
            Vjj = K.mean(
                centred_squared_j,
                axis=reduction_axes
            ) + self.epsilon
            Vkk = K.mean(
                centred_squared_k,
                axis=reduction_axes
            ) + self.epsilon
            Vri = K.mean(
                centred_r * centred_i,
                axis=reduction_axes,
            )
            Vrj = K.mean(
                centred_r * centred_j,
                axis=reduction_axes,
            )
            Vrk = K.mean(
                centred_r * centred_k,
                axis=reduction_axes,
            )
            Vij = K.mean(
                centred_i * centred_j,
                axis=reduction_axes,
            )
            Vik = K.mean(
                centred_i * centred_k,
                axis=reduction_axes,
            )
            Vjk = K.mean(
                centred_j * centred_k,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vjj = None
            Vkk = None
            Vri = None
            Vrj = None
            Vrk = None
            Vij = None
            Vik = None
            Vjk = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        input_bn = QuaternionBN(
            input_centred,
            Vrr, Vri, Vrj, Vrk, Vii,
            Vij, Vik, Vjj, Vjk, Vkk,
            self.beta,
            self.gamma_rr, self.gamma_ri,
            self.gamma_rj, self.gamma_rk,
            self.gamma_ii, self.gamma_ij,
            self.gamma_ik, self.gamma_jj,
            self.gamma_jk, self.gamma_kk,
            self.scale, self.center,
            axis=self.axis
        )
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean, mu, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vjj, Vjj, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vkk, Vkk, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vrj, Vrj, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vrk, Vrk, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vij, Vij, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vik, Vik, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vjk, Vjk, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centred = inputs - K.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = inputs
                return QuaternionBN(
                    inference_centred,
                    self.moving_Vrr, self.moving_Vri,
                    self.moving_Vrj, self.moving_Vrk,
                    self.moving_Vii, self.moving_Vij,
                    self.moving_Vik, self.moving_Vjj,
                    self.moving_Vjk, self.moving_Vkk,
                    self.beta,
                    self.gamma_rr, self.gamma_ri,
                    self.gamma_rj, self.gamma_rk,
                    self.gamma_ii, self.gamma_ij,
                    self.gamma_ik, self.gamma_jj,
                    self.gamma_jk, self.gamma_kk,
                    self.scale, self.center, axis=self.axis
                )

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(input_bn,
                                normalize_inference,
                                training=training)

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
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(QuaternionBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
