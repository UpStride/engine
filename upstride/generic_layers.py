"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
import functools
import inspect
from typing import List, Tuple
import tensorflow as tf

from .convolutional import Conv2D as Conv2DConj

# Definition of the GA, setup when upstride.type{1/2/3}.calling tf.keras.layers
upstride_type = 3
blade_indexes = ["", "1", "2", "3", "12", "13", "23", "123"]
geometrical_def = (3, 0, 0)

conjugate = False


def change_upstride_type(type, new_blade_indexes,  new_geometrical_def):
    global upstride_type, blade_indexes, geometrical_def
    upstride_type = type
    blade_indexes = new_blade_indexes
    geometrical_def = new_geometrical_def

def set_conjugaison_mult(b):
    global conjugate
    conjugate = b


def multivector_length() -> int:
    """map the upstride type to the number of dimensions in our GA
    """
    return len(blade_indexes)


def blade_index_to_position(index: str) -> int:
    @functools.lru_cache(maxsize=1)
    def get_dict():
        """return a dictionary that map the blade index to the position in the list encoding the multivector
        """
        d = {}
        for i, e in enumerate(blade_indexes):
            d[e] = i
        return d
    return get_dict()[index]


def square_vector(index: int) -> int:
    @functools.lru_cache(maxsize=1)
    def get_list():
        """return a list that map the indice to the sqare
        """
        l = [0]
        possible_squares = [1, -1, 0]
        for i in range(3):
            l += [possible_squares[i]] * geometrical_def[i]
        return l
    return get_list()[index]


def _ga_multiply_get_index(index_1: str, index_2: str) -> Tuple[int, str]:
    """given e_{index_1}, e_{index_2} return (s, index) such as e_{index_1} * e_{index_2} = s * e_{index}
    """
    l1 = [int(i) for i in index_1]
    l2 = [int(i) for i in index_2]
    s = 1

    # as l1 and l2 are already sorted, we can just merge them and count the number of permutation needed
    i1, i2, length_l1 = 0, 0, len(l1)
    out_l = []
    while i1 < len(l1) and i2 < len(l2):
        if l1[i1] == l2[i2]:
            # then move the element of l2 near the element of l1 and remove them
            if (length_l1 - 1) % 2 != 0:
                s *= -1
            # check the sign of the square
            s *= square_vector(l1[i1])
            length_l1 -= 1
            i1 += 1
            i2 += 1
        elif l1[i1] > l2[i2]:
            # then move the element of l2 before the element of l1
            if length_l1 % 2 != 0:
                s *= -1
            out_l.append(l2[i2])
            i2 += 1
        elif l1[i1] < l2[i2]:
            out_l.append(l1[i1])
            length_l1 -= 1
            i1 += 1
    out_l += l1[i1:] + l2[i2:]

    return s, "".join([str(i) for i in out_l])


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
    index1 = blade_indexes[i]
    index2 = blade_indexes[j]
    s, index = _ga_multiply_get_index(index1, index2)
    return blade_index_to_position(index), s


def get_layers(layer: tf.keras.layers.Layer, conj_layer: tf.keras.layers.Layer = None, *argv, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
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
    if 'name' not in kwargs or kwargs['name'] is None:
        layers = [layer(**kwargs) for _ in range(multivector_length())]
    else:
        layers = []
        base_name = kwargs['name']
        for i in range(multivector_length()):
            kwargs['name'] = f'{base_name}_{i}'
            layers.append(layer(**kwargs))

    if conj_layer is not None:
        kwargs['ga_dimension'] = multivector_length()
        conj_layer = conj_layer(**kwargs)
        return layers, add_bias, bias_parameters, conj_layer
    else:
        return layers, add_bias, bias_parameters, None


def dagger_sign() -> List[int]:
    """ return a array s such as for a vecor v of the GA, dagger{v} = s*v
    dagger is defined by dagger{v} * v = 1
    """
    s = []
    for i in blade_indexes:
        if len(i) in [0, 1]:
            s.append(1)
        else:
            s.append(-1)
    return s


def compute_all_cross_product(layers, inputs, convert_to_tf):
    layers_outputs = []
    if not convert_to_tf:
        # if there is no chance to convert back to tf, we can use time-distribute layer to speed up and save memory
        tdlayers = [tf.keras.layers.TimeDistributed(layer) for layer in layers]
        reshaped_inputs = [tf.keras.layers.Reshape([1] + e.shape[1:])(e) for e in inputs]
        inputs = tf.keras.layers.Concatenate(axis=1)(reshaped_inputs)
        td_outputs = [tdlater(inputs) for tdlater in tdlayers]

        # print(td_outputs[0].shape)

        for i in range(multivector_length()):
            layers_outputs.append([])
            for j in range(multivector_length()):
                layers_outputs[i].append(td_outputs[i][:, j, :])
    if convert_to_tf:
        # if there is a chance to convert back to tf, keep the simple way so tf will be able to prune ops
        for i in range(multivector_length()):
            layers_outputs.append([])
            for j in range(multivector_length()):
                layers_outputs[i].append(layers[i](inputs[j]))
    return layers_outputs


def geometric_multiplication(cross_product_matrix, inverse=False):
    output = [None] * multivector_length()
    for i in range(multivector_length()):
        for j in range(multivector_length()):
            if not inverse:
                k, s = unit_multiplier(i, j)
            else:
                k, s = unit_multiplier(j, i)

            # same as output[k] += s*self.layers[i](inputs[j]), but cleaner graph
            if s == 1:
                if output[k] is None:
                    output[k] = cross_product_matrix[i][j]
                else:
                    output[k] += cross_product_matrix[i][j]
            elif s == -1:
                if output[k] is None:
                    output[k] = -cross_product_matrix[i][j]
                else:
                    output[k] -= cross_product_matrix[i][j]
    return output


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
    def __init__(self, layer, *argv, upstride2tf=False, conj_layer=None, **kwargs):
        # if the layer can run conjugaison, then self.conj_layer is an instance of the conj layer, else none
        self.layers, self.add_bias, self.bias_parameters, self.conj_layer = get_layers(layer, conj_layer, *argv, **kwargs)
        self.convert_to_tf = upstride2tf

    def __call__(self, inputs):
        if not conjugate or self.conj_layer is None:
            if len(inputs) == 1:
                # real input
                x = inputs[0]
                output = [self.layers[i](x) for i in range(multivector_length())]
            else:
                # R^{multivector_length()} input
                layers_outputs = compute_all_cross_product(self.layers, inputs, self.convert_to_tf and not conjugate)
                output = geometric_multiplication(layers_outputs)
        else:
            outputs = self.conj_layer(inputs)
            print(outputs)
            # now outputs is
            # - a list of 3 dimension if inputs is a Tensor with outputs[i][j][0] = conv2D(pointwise_mult(w_i, w_j), x)
            # - a list of 3 dimension if inputs is a list with outputs[i][j][k] = conv2D(pointwise_mult(w_i, w_j), x_k)

            # now sum the results
            output = [None] * multivector_length()
            d = dagger_sign()
            for i in range(multivector_length()):
                for j in range(multivector_length()):
                    for k in range(len(outputs[0][0])):
                        t, s1 = unit_multiplier(i, k)
                        t, s2 = unit_multiplier(t, j)
                        if output[t] is None:
                            print(d)
                            print(outputs[i][j][k])
                            output[t] = s1*s2*d[i] * outputs[i][j][k]
                        else:
                            output[t] += s1*s2*d[i] * outputs[i][j][k]

        if self.add_bias:
            for i in range(multivector_length()):
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
        self.layers, self.add_bias, self.bias_parameters, _ = get_layers(layer, None, *argv, **kwargs)
        self.list_as_input = False  # some layers like Add or Contatenates takes a list of tensor as input

    def __call__(self, inputs):
        if self.list_as_input:
            inputs = reorder(inputs)

        if len(inputs) == 1:
            output = [self.layers[i](inputs[0]) for i in range(multivector_length())]
        else:
            output = [self.layers[i](inputs[i]) for i in range(multivector_length())]
        return output


class Conv2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Conv2D, *argv, conj_layer=Conv2DConj, **kwargs)


class Dense(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Dense, *argv, **kwargs)


class Conv2DTranspose(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Conv2DTranspose, *argv, **kwargs)


class UpSampling2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.UpSampling2D, *argv, **kwargs)


class DepthwiseConv2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.DepthwiseConv2D, *argv, **kwargs)


class DepthwiseConv2DTranspose(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.DepthwiseConv2DTranspose, *argv, **kwargs)


class SeparableConv2D(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.SeparableConv2D, *argv, **kwargs)


class MaxPooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.MaxPooling2D, *argv, **kwargs)


class AveragePooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.AveragePooling2D, *argv, **kwargs)


class MaxPool2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.MaxPool2D, *argv, **kwargs)


class AveragePool2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.AveragePool2D, *argv, **kwargs)


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


class Cropping2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Cropping2D, *argv, **kwargs)


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


class Multiply(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Multiply, *argv, **kwargs)
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
