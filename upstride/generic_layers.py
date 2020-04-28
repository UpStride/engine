"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
from typing import Tuple, List
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


def unit_multiplier(i, j):
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


class GenericLinear:
    def __init__(self, layer, *argv, **kwargs):
        self.layers = [layer(*argv, **kwargs) for _ in range(type_to_multivector_length[upstride_type])]

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
        return output


class GenericNonLinear:
    def __init__(self, layer, *argv, **kwargs):
        self.layers = [layer(*argv, **kwargs) for _ in range(type_to_multivector_length[upstride_type])]

    def __call__(self, inputs):
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


class GlobalMaxPooling2D(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.GlobalMaxPooling2D, *argv, **kwargs)


class Reshape(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Reshape, *argv, **kwargs)


class BatchNormalization(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.BatchNormalization, *argv, **kwargs)


class Activation(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Activation, *argv, **kwargs)


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
