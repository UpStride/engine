"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
import inspect
from typing import List, Tuple
import tensorflow as tf

upstride_type = 3  # setup when calling upstride.type{1/2/3}

type_to_multivector_length = {
    1: 2,
    2: 4,
    3: 8
}

type3_multivector_to_index = ["", "1", "2", "3", "12", "13", "23", "123"]
type3_index_to_multivector = {
    "": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "12": 4,
    "13": 5,
    "23": 6,
    "123": 7
}


def _ga_multiply_get_index(index_1: str, index_2: str) -> Tuple[int, str]:
    """given e_{index_1}, e_{index_2} return (s, index) such as e_{index_1} * e_{index_2} = s * e_{index}
    """
    even_number_of_permutations = True
    index = [int(i) for i in index_1 + index_2]
    # first we sort the index
    for i in range(len(index) - 1):
        for j in range(len(index) - i - 1):
            if int(index[j]) > int(index[j + 1]):
                index[j], index[j + 1] = index[j + 1], index[j]
                even_number_of_permutations = not even_number_of_permutations
    # then we remove the doubles
    i = 0
    while i < len(index) - 1:
        if index[i] == index[i + 1]:
            index = index[:i] + index[i+2:]
        else:
            i += 1
        # TODO for generic GA, we need to know the parameters of G(a,b,c), to know if the square is 1, -1, or 0
        # one solution : add a,b and c as global variables and compare index[i] with a, a+b, a+b+c
    return even_number_of_permutations, "".join([str(i) for i in index])


def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
    """given e_i and e_j, return (k,s) such as : e_i * e_j = s * e_k

    with:
        e_0 = 1, e_1 = i if upstride_type == 1
        e_0 = 1, e_1 = i, e_2 = j, e_3 = k if upstride_type == 2
        s in {-1, 1}

    for instance, upstride_type == 1,
    (0, 0) -> (0, 1) because e_0 * e_0 = 1 * 1 = 1 * e_0
    (0, 1) -> (1, 1) because e_0 * e_1 = 1 * e_1
    (1, 1) -> (0, -1) because e_1 * e_1 = i**2 = -1 = -1 * e_0
    """
    # rules are the same for complex and quaternion
    if upstride_type in [1, 2]:
        if i == 0 and j == 0:  # real multiplies by real gives real
            return 0, 1
        elif i == j:
            return 0, -1
        elif i == 0 or j == 0:
            return max(i, j), 1
        else:  # this case is only for quaternion
            id = 6 - i - j
            if i > j:
                return id, -1
            if i < j:
                return id, 1
    if upstride_type == 3:  # (scalar, e1, e2, e3, e12, e13, e23, e123)
        index1 = type3_multivector_to_index[i]
        index2 = type3_multivector_to_index[j]
        even_number_of_permutations, index = _ga_multiply_get_index(index1, index2)
        if even_number_of_permutations:
            s = 1
        else:
            s = -1
        return type3_index_to_multivector[index], s


def get_layers(layer: tf.keras.layers.Layer, *argv, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
    """instantiate layer several times to match the number needed by the GA definition

    Any parameter analysis need to be done here. For instance, we can't define several times 
    a layer with the same name, so we need to edit the name manually

    Args:
        layer (tf.keras.layers.Layer): a keras layer that we need to instantiate several times

    Returns:
        List[tf.keras.layers.Layer]: the list of keras layers
    """
    # convert all arguments from argv to kwargs
    parameters = inspect.getfullargspec(layer.__init__).args
    for i, arg in enumerate(argv):
        kwargs[parameters[i + 1]] = arg  # + 1 because the first element of parameters is 'self'
    # add all default parameters to kwargs
    for key, value in inspect.signature(layer.__init__).parameters.items():
        if key in ['self', 'kwargs']:
            continue
        if key not in kwargs:
            kwargs[key] = value.default

    # If we define some bias, we don't want to put it in the linear layer but after, as a non-linear layer
    add_bias = False
    if "use_bias" in kwargs:
        add_bias = kwargs["use_bias"]
        kwargs["use_bias"] = False
    bias_parameters = {}
    if add_bias:
        for param in ["bias_initializer", "bias_regularizer", "bias_constraint"]:
            bias_parameters[param] = kwargs[param]
            
    # special case for the name of the layer : if defined, then we need to change it to create different operations
    if 'name' not in kwargs:
        layers = [layer(**kwargs) for _ in range(type_to_multivector_length[upstride_type])]
    else:
        layers = []
        base_name = kwargs['name']
        for i in range(type_to_multivector_length[upstride_type]):
            kwargs['name'] = f'{base_name}_{i}'
            layers.append(layer(**kwargs))
    return layers, add_bias, bias_parameters


class BiasLayer(tf.keras.layers.Layer):
    """Keras layer that only adds a bias to the input.

    code from https://github.com/tensorflow/agents/blob/v0.4.0/tf_agents/networks/bias_layer.py#L24-L81
    with some modifications when initializing the weight to use the same conf as other layers

    `BiasLayer` implements the operation:
    `output = input + bias`
    Arguments:
        bias_initializer: Initializer for the bias vector.
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`. The most common
          situation would be a 2D input with shape `(batch_size, input_dim)`. Note
          a rank of at least 2 is required.
    Output shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`. For instance, for a
          2D input with shape `(batch_size, input_dim)`, the output would have
          shape `(batch_size, input_dim)`.
    """

    def __init__(self, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(BiasLayer, self).__init__(**kwargs)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])

        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `BiasLayer` '
                             'should be defined. Found `None`.')

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.bias = self.add_weight(
            name='bias',
            shape=[input_shape[-1]],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)

        self.built = True

    def call(self, inputs):
        return tf.nn.bias_add(inputs, self.bias)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'bias_initializer':
                tf.keras.initializers.serialize(self.bias_initializer),
        }
        base_config = super(BiasLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GenericLinear:
    def __init__(self, layer, *argv, **kwargs):
        self.layers, self.add_bias, self.bias_parameters = get_layers(layer, *argv, **kwargs)

    def __call__(self, inputs):
        if len(inputs) == 1:
            # real input
            x = inputs[0]
            output = [self.layers[i](x) for i in range(type_to_multivector_length[upstride_type])]
        else:
            # R^{type_to_multivector_length[upstride_type]} input
            output = [None] * type_to_multivector_length[upstride_type]
            for i in range(type_to_multivector_length[upstride_type]):
                for j in range(type_to_multivector_length[upstride_type]):
                    k, s = unit_multiplier(i, j)
                    # same as output[k] += s*self.layers[i](inputs[j]), but cleaner graph
                    if s == 1:
                        if output[k] is None:
                            output[k] = self.layers[i](inputs[j])
                        else:
                            output[k] += self.layers[i](inputs[j])
                    else:
                        if output[k] is None:
                            output[k] = -self.layers[i](inputs[j])
                        else:
                            output[k] -= self.layers[i](inputs[j])
        if self.add_bias:
            for i in range(type_to_multivector_length[upstride_type]):
                output[i] = BiasLayer(self.bias_parameters['bias_initializer'], self.bias_parameters['bias_regularizer'], self.bias_parameters['bias_constraint'])(output[i])
        return output


def reorder(inputs):
    # need to permute the 2 dimensions of the list
    # for instance, if layer is Add on quaternion, inputs = [[a,b,c,d], [e,f,g,h]]
    # need to transform to [[a,e],[b,f],[c,g],[d,h]]
    new_inputs = [[] for _ in range(len(inputs[1]))]
    for el in inputs:
        for i, e in enumerate(el):
            new_inputs[i].append(e)
    return new_inputs


class GenericNonLinear:
    def __init__(self, layer, *argv, **kwargs):
        self.layers, self.add_bias, self.bias_parameters = get_layers(layer, *argv, **kwargs)
        self.list_as_input = False  # some layers like Add or Contatenates takes a list of tensor as input

    def __call__(self, inputs):
        if self.list_as_input:
            inputs = reorder(inputs)

        if len(inputs) == 1:
            output = [self.layers[i](inputs[0]) for i in range(type_to_multivector_length[upstride_type])]
        else:
            output = [self.layers[i](inputs[i]) for i in range(type_to_multivector_length[upstride_type])]
        return output


class Conv2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Conv2D, *argv, **kwargs)


class Dense(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Dense, *argv, **kwargs)


class Conv2DTranspose(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Conv2DTranspose, *argv, **kwargs)


class DepthwiseConv2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.DepthwiseConv2D, *argv, **kwargs)


class DepthwiseConv2DTranspose(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.DepthwiseConv2DTranspose, *argv, **kwargs)


class MaxPooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.MaxPooling2D, *argv, **kwargs)


class MaxPool2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.MaxPool2D, *argv, **kwargs)


class GlobalMaxPooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.GlobalMaxPooling2D, *argv, **kwargs)


class GlobalAveragePooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.GlobalAveragePooling2D, *argv, **kwargs)


class Reshape(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Reshape, *argv, **kwargs)


class BatchNormalization(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.BatchNormalization, *argv, **kwargs)


class Activation(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Activation, *argv, **kwargs)


class Flatten(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Flatten, *argv, **kwargs)


class ZeroPadding2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.ZeroPadding2D, *argv, **kwargs)


class ReLU(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.ReLU, *argv, **kwargs)


class LeakyReLU(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.LeakyReLU, *argv, **kwargs)


class Add(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Add, *argv, **kwargs)
        self.list_as_input = True


class Concatenate(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Concatenate, *argv, **kwargs)
        self.list_as_input = True


class Dropout(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        # TODO for now dropout manage real part and img part separately. better if manage both at the same time
        # solution : only one layer and concat before ? Or define all Dropout with the same seed ?
        super().__init__(tf.keras.layers.Dropout, *argv, **kwargs)


class TF2Upstride:
    """for compatibility with the c++ version. convert a tensor to a list of length one of tensor
    the list will have the good size after the first operation
    """

    def __call__(self, x):
        return [x]


class Upstride2TF:
    """convert multivector back to real values. 
    """

    def __init__(self, strategy='default'):
        # for now strategy is useless
        if strategy != 'default':
            raise NotImplementedError("")

    def __call__(self, x):
        return x[0]
