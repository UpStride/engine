"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""

import tensorflow as tf

upstride_type = 2  # setup when calling upstride.type{1/2/3}

type_to_multivector_length = {
    1: 2,
    2: 4,
    3: 8
}


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
        raise NotImplementedError("")
        if i == j and i < 4:
            return 0, 1
        elif i == j:
            return 0, -1
        elif i == 0 or j == 0:
            return max(i, j), 1


class GenericLinear:
    def __init__(self, layer, *argv, **kwargs):
        self.layers = [layer(*argv, **kwargs) for _ in range(type_to_multivector_length[upstride_type])]

    def __call__(self, inputs):
        if len(inputs) == 1:
            # real input
            x = inputs[0]
            output = [self.layers[i](x) for i in range(2**upstride_type)]
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
        pass

    def __call__(self, x):
        return x[0]
