import tensorflow as tf
from .... import generic_layers
from ....generic_layers import *
from .activations import *
from ....batchnorm import BatchNormalizationC

generic_layers.upstride_type = 1
generic_layers.blade_indexes = ["", "12"]
generic_layers.geometrical_def = (2, 0, 0)

# If you wish to overwrite some layers, please implements them here


class Conv2D(tf.keras.Model):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.Conv2D.__init__, argv, kwargs)
    kwargs, self.add_bias, bias_parameters = remove_bias_from_kwargs(kwargs)
    # TODO this need to be cleaned, have a look at TF code to see how they do
    if 'name' in kwargs:
      super().__init__(name=kwargs['name'])
    else:
      super().__init__()

    # save the parameter so separating real and complex part is easier
    self.filters = kwargs['filters']

    if kwargs['data_format'] is not None:
      self.data_format = kwargs['data_format']
    else:
      self.data_format = tf.keras.backend.image_data_format()

    # this convolution will do the multiplication by real and imaginary parts at the same time
    kwargs['filters'] = 2 * kwargs['filters']
    self.conv = tf.keras.layers.Conv2D(**kwargs)

    # define the bias layer
    if self.add_bias:
      real_bias = BiasLayer(bias_parameters['bias_initializer'], bias_parameters['bias_regularizer'], bias_parameters['bias_constraint'])
      img_bias = BiasLayer(bias_parameters['bias_initializer'], bias_parameters['bias_regularizer'], bias_parameters['bias_constraint'])
      self.bias = [real_bias, img_bias]

  def build(self, input_shape):
    """
    Args:
      input_shape: list of TensorShape, we need to extract the first one to build the convolution
    """
    return self.conv.build(input_shape[0])

  def split_conv_output(self, output):
    """ Split the output of the convolution to real and img part
    Is data_format aware
    """
    if self.data_format == 'channels_last':
      output_real, output_img = output[:, :, :, :self.filters], output[:, :, :, self.filters:]
    else:
      output_real, output_img = output[:, :self.filters, :, :], output[:, self.filters:, :, :]
    return output_real, output_img

  def call(self, input):
    """
    Args:
      input: list of 2 Tensors [real_part, img_part]

    Returns:
      list of 2 Tensors [real_part, img_part]
    """
    input_real = input[0]
    input_img = input[1]

    output_real_real, output_real_img = self.split_conv_output(self.conv(input_real))
    output_img_real, output_img_img = self.split_conv_output(self.conv(input_img))
    output = [output_real_real - output_img_img, output_real_img + output_img_real]

    if self.add_bias:
      output = [self.bias[i](output[i]) for i in range(2)]
    return output
