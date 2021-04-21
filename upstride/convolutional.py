import tensorflow as tf
from .uptypes_utilities import prepare_inputs, prepare_output, prepare_hyper_weight


class Conv2DParcollet(tf.keras.layers.Conv2D):
  def __init__(self, uptype, *args, **kwargs):
    self.uptype = uptype # FIXME generalize when implementing for other uptypes
    self.axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    super().__init__(*args, **kwargs)

  def add_weight(self, **kwargs):
    # Intercepting add_weight to inject the extra dimension needed by upstride datatypes.
    kwargs['shape'] = (self.uptype.multivector_length, *kwargs['shape'])
    return super().add_weight(**kwargs)

  def call(self, inputs):
    if self._is_causal:  # Apply causal padding to inputs for Conv1D.
      inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

    axis = 1 if self.axis == 1 else tf.rank(inputs)
    kernel = prepare_hyper_weight(self.uptype, self.kernel, groups=self.groups)
    inputs = prepare_inputs(self.uptype, inputs, channels_axis=axis, groups=self.groups)
    outputs = self._convolution_op(inputs, kernel)

    if self.use_bias: # bias.shape (N, O)
      output_rank = outputs.shape.rank
      # If it is a grouped convolution, then the bias needs to be reshaped from [N*O] to [O*N],
      # to be consistent with the shape in the variable outputs. For details, go to the definition
      # of prepare_inputs()
      bias = self.bias
      if self.groups > 1:
        bias = tf.transpose(bias, perm=[1, 0]) # shape [O, N]
        bias = tf.reshape(bias, [-1]) # shape [O*N]
      else:
        bias = tf.reshape(self.bias, -1) # shape [N*O]
      if self.rank == 1 and self._channels_first:
        # nn.bias_add does not accept a 1D input tensor.
        bias = array_ops.reshape(bias, (1, self.uptype.multivector_length * self.filters, 1))
        outputs += bias
      else:
        # Handle multiple batch dimensions.
        if output_rank is not None and output_rank > 2 + self.rank:

          def _apply_fn(o):
            return tf.nn.bias_add(o, bias, data_format=self._tf_data_format)

          outputs = nn_ops.squeeze_batch_dims(
              outputs, _apply_fn, inner_rank=self.rank + 1)
        else:
          outputs = tf.nn.bias_add(
              outputs, bias, data_format=self._tf_data_format)

    outputs = prepare_output(self.uptype, outputs, channels_axis=axis, groups=self.groups)
    if self.activation is not None:
      return self.activation(outputs)
    return outputs