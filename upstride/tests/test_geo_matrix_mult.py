import pytest
import tensorflow as tf
from upstride.tests.utility import assert_small_float_difference, uptypes, algebra_maps, create_components
from upstride import generic_layers

class TestGeometricMatrixMultiply:

    @pytest.mark.parametrize('uptype', ['up0', 'up1', 'up2'])
    @pytest.mark.parametrize('batch_size, d_extra, d1, d2, d3', [
        (1, 2, 2, 2, 2),
        (3, 6, 4, 5, 2),
        (7, 7, 7, 7, 7),
    ])
    def test_product(self, uptype, batch_size, d_extra, d1, d2, d3):
        uptype = 'up1'
        component_shape_t1 = (batch_size, d_extra, d1, d2)
        component_shape_t2 = (batch_size, d_extra, d2, d3)

        algebra_map = algebra_maps[uptype]
        hyper_dimension = uptypes[uptype].multivector_length

        t1 = create_components(hyper_dimension, component_shape_t1)
        t2 = create_components(hyper_dimension, component_shape_t2)

        layer = generic_layers.GeometricMatrixMultiply(uptypes[uptype])
        inp = [tf.concat(t1, axis=0), tf.concat(t2, axis=0)]
        test_out = layer(inp)

        ref_partial = [[] for _ in range(hyper_dimension)]
        for i in range(hyper_dimension):
            for j in range(hyper_dimension):
                inter_res = tf.matmul(t1[i], t2[j])
                ref_partial[i].append(inter_res)

        ref_outputs = [0 for _ in range(hyper_dimension)]
        for i in range(hyper_dimension):
            for j in range(hyper_dimension):
                which, coeff = algebra_map[i][j]
                ref_outputs[which] += ref_partial[i][j] * coeff

        ref_out = tf.concat(ref_outputs, axis=0)
        assert_small_float_difference(test_out, ref_out, 0.001)


    def simple_test_base(self, t1, t2, ref_out):
        layer = generic_layers.GeometricMatrixMultiply(uptypes['up1'])
        inp = [t1, t2]
        test_out = layer(inp)
        assert_small_float_difference(test_out, ref_out, 0.001)

    def test_simple_2x1x1(self):
        t1 = [
            [[-4.],
            [-3.]],

            [[ 1.],
            [-2.]]
        ]
        t2 = [
            [[2.]],

            [[3.]]
        ]
        ref_out = [
            [[-11.],
            [  0.]],

            [[-10.],
            [-13.]]
        ]
        self.simple_test_base(t1, t2, ref_out)

    def test_simple_1x1x2(self):
        t1 = [
            [[-3.]],

            [[-1.]]
        ]
        t2 = [
            [[-3., -4.]],

            [[ 1., -2.]]
        ]
        ref_out = [
            [[10., 10.]],

            [[ 0., 10.]]
        ]
        self.simple_test_base(t1, t2, ref_out)

    def test_simple_1x2x1(self):
        t1 = [
            [[-3., -4.]],

            [[ 3.,  2.]]
        ]
        t2 = [
            [[-1.],
            [ 1.]],

            [[-3.],
            [-2.]]
        ]
        ref_out = [
            [[12.]],

            [[16.]]
        ]
        self.simple_test_base(t1, t2, ref_out)

    def test_simple_2x2x2(self):
        t1 = [
            [[ 1., -4.],
            [-1., -2.]],

            [[-3.,  3.],
             [-4.,  1.]]
        ]
        t2 = [
            [[ 2.,  3.],
            [ 2.,  2.]],

            [[ 2.,  2.],
            [-2., -2.]]
        ]
        ref_out = [
            [[ 6.,  7.],
            [ 4.,  3.]],

            [[10.,  7.],
            [-4., -8.]]
        ]
        self.simple_test_base(t1, t2, ref_out)