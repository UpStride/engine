import unittest
import numpy as np
from math import pi
from activations import *

##TODO: add the learnable parameter and initialize it properly in TF

def activation_pow2(z):
    """
    Activation function for complex numbers z=a+ib
    Forward pass of the activation function: F(z)=z^2
    We can rewrite F(z) as F(a+ib)=[a^2-b^2]+i[2ab]
    """

    a, b = z[0], z[1]
    Re_F = power(a,2)-power(b,2)
    Im_F = 2*multiply(a,b)
    
    return [Re_F, Im_F]

def grad_activation_pow2(z):
    """
    Backward pass (gradient) of the activation function: F(z)=z^2, with z=a+ib
    """

    a, b = z[0], z[1]
    gradF_a = 2*a+2*b
    gradF_b = 2*a-2*b

    return [gradF_a, gradF_b]
    

class TestActivationCos(unittest.TestCase):

    def test_forward_scalar(self):
        a, b = 0, 0
        target_a, target_b = 1, 0
        out = activation_cos([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_forward_zeros(self):
        a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
        target_a, target_b = np.ones([2,3,4]), np.zeros([2,3,4])
        out = activation_cos([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_forward_90deg(self):
        a, b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
        target_a, target_b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
        out = activation_cos([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_backward_zeros(self):
        a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
        target_a, target_b = np.ones([2,3,4]), np.ones([2,3,4])
        out = grad_activation_cos([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_backward_90deg(self):
        a, b = np.ones([2,3,4])*(pi/2), np.zeros([2,3,4])
        target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])
        out = grad_activation_cos([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b)) 
        
class TestActivationPow2(unittest.TestCase):

    def test_forward_scalar(self):
        a, b = 0, 0
        target_a, target_b = 0, 0
        out = activation_pow2([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_forward_zeros(self):
        a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
        target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])
        out = activation_pow2([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_forward_ones(self):
        a, b = np.ones([2,3,4]), np.ones([2,3,4])
        target_a, target_b = np.zeros([2,3,4]), 2*np.ones([2,3,4])
        out = activation_pow2([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_backward_zeros(self):
        a, b = np.zeros([2,3,4]), np.zeros([2,3,4])
        target_a, target_b = np.zeros([2,3,4]), np.zeros([2,3,4])
        out = grad_activation_pow2([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b))

    def test_backward_ones(self):
        a, b = np.ones([2,3,4]), np.ones([2,3,4])
        target_a, target_b = 4*np.ones([2,3,4]), np.zeros([2,3,4])
        out = grad_activation_pow2([a,b])
        self.assertTrue(np.array_equal(out[0],target_a) and np.array_equal(out[1],target_b)) 