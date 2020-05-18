---
title: Upstride engine in python implementing $\mathbb{G} \circ \mathbb{M}$
author:
date: \today
---

\newpage
# Introduction
This code implement Geometric Algebra in TensorFlow using Keras high-level API. It is compatible with latest release of tf 1 ($>=1.13$) and tf $2$.

This document explains how to implement Geometric Algebra using TensorFlow.

# Linear layer

This section describes how to implement *any* linear layer in TensorFlow, for instance:

  - Conv2D
  - Dense
  - Conv2DTranspose
  - DepthwiseConv2D
  - DepthwiseConv2DTranspose

## On $\mathbb{R}$
For now, lets take a look at linear layer on $\mathbb{R}$. Let:

  - $a$ the input vector of shape $n$
  - $b$ the output vector of shape $m$
  - $M$ the linear transformation with $m$ row and $n$ column
  
Then computing this layer means to compute the product : $b = Ma$, and in term of TensorFlow, this leads to this topology:

![real multiplication](./doc/model_tf.png "real multiplication"){width=70%}


## On $\mathbb{C}$
Now, if we work on complex number, then 

  - $a = a_R + ia_C$
  - $b = b_R + ib_C$
  - $M = M_R + iM_C$
  
Then computing this layer means to compute the product : $b = Ma$, or

  - $b_R = M_Ra_R - M_Ca_C$
  - $b_C = M_Ra_C + M_Ca_R$

and in term of TensorFlow, this leads to this topology:

![complex multiplication](./doc/complex_mult.svg "complex multiplication"){width=100%}

## On any geometical algebra
Now, lets work with generic geometical algebra, defined by a set of blades $(e_i)_{i\in[0, n]}$ that can be scalar, vector, bi-vector or multi-vectors.

  - $a = \sum_{i=0}^{n}a_ie_i$
  - $b = \sum_{i=0}^{n}b_ie_i$
  - $M = \sum_{i=0}^{n}M_ie_i$

Then

$$b = \sum_{i=0}^{n}\sum_{j=0}^{n}M_ia_je_ie_j$$

the product $e_ie_j$ is defined by the structure of the GA, but in orthogonal cases, it gives : $e_ie_j = se_k$ with $k\in[0,n]$ and $s\in\{-1, 0, 1\}$

This formula allows us to implement all linear layer for all geometical algebra at the same time.

## Python implementation

No more mathematics. Now we can code!

so first, we need this fonction that compute the product of two blades, and return a blade and $s$. In the python code this is the function 

```python
def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
    """given e_i and e_j, return (k,s) such as : e_i * e_j = s * e_k
    """
```

Then, the idea is to define a class `GenericLinear` that we will be able to inherit to easely define all our linear layer in a very simple way.
For instance, for a Dense layer: 

```python
class Dense(GenericLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Dense, *argv, **kwargs)
```

the `__init__` method of the Dense layer takes exactly the same arguments as the Keras Dense layer, but 
the `__init__` method of the `GenericLinear` class take one more argument: the keras class to call (`tf.keras.layers.Dense` here)

the `GenericLinear` class initialize the Keras layer once for every dimension of our GA, and then, when called, 2 `for` loops compute all the products between
all the blade to compute the output

### The bias problem

In deep learning, we often add a bias term after a linear layer. In Keras, this bias is handle by the linear layer, which is a problem here.

Indeed, let's have a look at $\mathbb{C}$. If the bias is in the linear layer then the operation we do are:


$$o_R = M_R a_R - M_C a_C + b_R - b_C$$
$$o_C = M_R a_C + M_C a_R + b_R + b_C$$

In the middle of the neural network, we have 2 parameters $(b_R, b_C)$ and 2 constraints, so it shouldn't hurt the training (even if it is not efficient because we have 2 more operation than needed).
But at the end of the neural network, as we only keep the real part, we only have one constraint and this can hurt the performances. So we need to prevent this 
from happening.

One solution is to detect when the user ask for a bias, remove it from the linear part and add in after the linear layer.
The easiest way to do it in python is using inspection. 

```python
# convert all arguments from argv to kwargs
parameters = inspect.getfullargspec(layer.__init__).args
for i, arg in enumerate(argv):
    kwargs[parameters[i + 1]] = arg  
    # + 1 because the first element of parameters is 'self'
# add all default parameters to kwargs
for key, value in inspect.signature(layer.__init__).parameters.items():
    if key in ['self', 'kwargs']:
        continue
    if key not in kwargs:
        kwargs[key] = value.default

# If we define some bias, save its parameters
add_bias = False
if "use_bias" in kwargs:
    add_bias = kwargs["use_bias"]
    kwargs["use_bias"] = False
bias_parameters = {}
if add_bias:
    for param in ["bias_initializer", "bias_regularizer", "bias_constraint"]:
        bias_parameters[param] = kwargs[param]
# now we can add the linear layer, then the bias
# ......
```

# Non-linear layers

non linear layers are for instance :

  - MaxPooling2D
  - GlobalAveragePooling2D
  - Reshape
  - BatchNormalization
  - Activation
  - Flatten
  - ZeroPadding2D
  - Add
  - Concatenate

The idea here is again to define a class `GenericNonLinear` that we will be able to inherit to  define most of our non-linear layer (except some of them),
the same way as our linear layer. For instance:

```python
class Activation(GenericNonLinear):
    def __init__(self, *argv, **kwargs):
        super().__init__(tf.keras.layers.Activation, *argv, **kwargs)
```

For most of the non-linear layer, the equation is simply:

$$o = \sum_{i=0}^{n}f_i(a_i)e_i$$

with

  - $o$ the output
  - $a$ the input
  - $e_i$ the blades
  - $f_i$ the non-linear function (for instance $f(x) = max(0, x)$)

the `GenericNonLinear` class initialize the Keras layer once for every dimension of our GA, and when called, compute the output with a simple loops

## Non-linear with many inputs

most of the non-linear layers have only one input, but some can have several (Add and Concatenate).
The implementation is the same, we just need to reorder the inputs before computation. 
Indeed, these keras layers takes a list of Tensor as input $[T_1, T_2]$

when working on GA, or input is a list of list of Tensor. The length of the second list 
is the number of component in the GA, so if working with quaternion, the input will be
$[[T_{11},T_{12},T_{13},T_{14}], [T_{21},T_{22},T_{23},T_{24}]]$

If we reorder the list this way: $[[T_{11},T_{21}], [T_{12},T_{22}], [T_{13},T_{23}], [T_{14},T_{24}]]$
, the rest of the implementation stay valid

## Exception

### Dropout
When performing dropout, there are several possible strategies: we can cancel weights on the several component of the multivector in a non-correlated way
or a correlated way. The current implementation can be both:

  - the non-correlated way is the default strategy
  - correlated way can be achieved if the user define a random seed when creating the layer. All the keras layers will have the same seed and so the
  same behaviour

### BatchNormalization

The current implementation works on the several component of the multivector in a non-correlated way. Maybe in the future we could do something clever ?


# Compatibility with the C++ engine

the C++ engine uses 2 layers to convert data between TensorFlow and Upstride : `TF2Upstride` and `Upstride2TF`
these 2 operations are not usefull with the python version but kept for combatibility.

Also the C++ engine doesn't define all the layers : some of the non linear for instance don't
need to be implemented. But we need them for the python version

One fix for now is to link the upstride version with the keras version at runtime, so it is transparent to the user.
This is implemented in the `imagenet_baseline` repository

```python
import upstride.type3.tf.keras.layers as up_layers
layers_to_register = ['Dropout', ...]
for l in layers_to_register:
    try:
        a = getattr(up_layers, l)
    except AttributeError as e:
        setattr(up_layers, l, getattr(tf.keras.layers, l))
```

# Futur improvement
  - better gradient computation
  - optimize specific implementation (for instance quaternion)
  - improve BatchNormalization


# Improvement from v0.1.1 to v0.1.2

## Generic Multiplication
To improve the backpropagation efficiency, a solution is to merge the several gradients before backpropagate to the linear layer.
TensorFlow provide a way to do it using the TimeDistribe Layer. The idea is to merge the several calls to an operation into one to enable some optimization.

![complex multiplication](./doc/complex_mult.svg "complex multiplication"){width=100%}

![complex multiplication](./doc/mult_0.1.1.png "complex multiplication"){width=100%}

## Quaternion multiplication

Given 2 quaternions $(a_1 + a_2i + a_3j + a_4k)$ and $(b_1 + b_2i + b_3j + b_4k)$, the naive way to compute the product $c$ is :

$$
\begin{cases} 
  c_1 = & a_1b_1 - a_2b_2 - a_3b_3 - a_4b_4 \\ 
  c_2 = & a_1b_2 + a_2b_1 + a_3b_4 - a_4b_3 \\ 
  c_3 = & a_1b_3 + a_3b_1 + a_4b_2 - a_2b_4 \\ 
  c_4 = & a_1b_4 + a_4b_1 + a_2b_3 - a_3b_2 \\ 
\end{cases}
$$

So 16 multiplications and 12 additions.

In term of tensorflow operations, because of the isomorphism between $\mathbb{M} \circ \mathbb{G}$ and $\mathbb{G} \circ \mathbb{M}$, it means for implementing a linear layer on $\mathbb{H}$, it takes 16 calls to the linear layer on $\mathbb{R}$

The same results can be achieved by computing: 

$$\begin{pmatrix} n_0 & n_1 &n_2 &n_3 \end{pmatrix}= \begin{pmatrix} a_0 & a_1 &a_2 &a_3 \end{pmatrix}A $$ 
$$\begin{pmatrix} p_0 & p_1 &p_2 &p_3 \end{pmatrix}= \begin{pmatrix} b_0 & b_1 &b_2 &b_3 \end{pmatrix}A $$ 

$$\begin{pmatrix}−c_0 & c_1 & c_2 & c_3\end{pmatrix}=0.25\begin{pmatrix}n_0p_0 & n_1p_1 & n_2p_2 & n_3p_3\end{pmatrix}A−2\begin{pmatrix}a_0b_0 & a_3b_2 & a_2b_1 & a_1b_3\end{pmatrix}$$

with 

$$ A = \begin{pmatrix} 1 & 1& 1& 1 \\ 1 & -1 &1&-1\\1&1&-1&-1\\1&-1&-1&1 \end{pmatrix}$$

We will call this method 1. 40 additions and 8 multiplications are needed, so for implementing a linear layer, 8 calls to the linear layer on $\mathbb{R}$ are needed.

Another method is to compute :

$$
\begin{cases} 
  A_1 = & (a_4 + a_2)(b_2 + b_3) \\
  A_3 = & (a_1 - a_3)(b_1 + b_4) \\
  A_4 = & (a_1 + a_3)(b_1 - b_4) \\
  A_2 = & A_1 + A_3 + A_4 \\
  A_5 = & 0.5(A_2 + (a_4-a_2)(b_2-b_3)) \\
  Q_1 = & A_5 - A_1 + (a_4 - a_3)(b_3 - b_4) \\
  Q_2 = & A_5 - A_2 + (a_2 + a_1)(b_2 + b_1) \\
  Q_3 = & A_5 - A_3 + (a_1 - a_2)(b_3 + b_4) \\
  Q_4 = & A_5 - A_4 + (a_4 + a_3)(b_1 - b_2) \\
\end{cases}
$$

This method takes allso 8 calls to the linear layer on $\mathbb{R}$ to implement the layer on $\mathbb{H}$.

After implementing these methods, we found that they work very well when working with float 32, but not in float 16.

One posibiilities is that the precision error with with computation method prevent it from working in float 16, so we neede to
do a precision computation

### Noise analysis on float 16
first lets found out the noise of the basic operations. To do so we generate 10000 random numbers, compute the product and addition in float 64, 32 and 16 and
compare the difference between the results

<!-- ```python
import tensorflow as tf
a = tf.random.uniform((10000,), maxval=1, dtype=tf.float64)
b = tf.random.uniform((10000,), maxval=1, dtype=tf.float64)
sum_64 = a+b
sum_32 = tf.cast(tf.cast(a, tf.float32)+tf.cast(b, tf.float32), tf.float64)
sum_16 = tf.cast(tf.cast(a, tf.float16)+tf.cast(b, tf.float16), tf.float64)
diff_sum_32 = tf.sort(tf.abs(sum_64 - sum_32))
diff_sum_16 = tf.sort(tf.abs(sum_64 - sum_16))
max_sum_32 = tf.math.reduce_max(diff_sum_32)
max_sum_16 = tf.math.reduce_max(diff_sum_16)
``` -->

The source code of these different experiments is available in the Readme

for additions,
  - max error on float 32 : $1.2 \times 10^{-7}$
  - max error on float 16 : $9.7 \times 10^{-4}$

the same work on multiplication give:
<!-- ```python
import tensorflow as tf
a = tf.random.uniform((10000,), maxval=1, dtype=tf.float64)
b = tf.random.uniform((10000,), maxval=1, dtype=tf.float64)
sum_64 = a*b
sum_32 = tf.cast(tf.cast(a, tf.float32)*tf.cast(b, tf.float32), tf.float64)
sum_16 = tf.cast(tf.cast(a, tf.float16)*tf.cast(b, tf.float16), tf.float64)
diff_sum_32 = tf.sort(tf.abs(sum_64 - sum_32))
diff_sum_16 = tf.sort(tf.abs(sum_64 - sum_16))
max_sum_32 = tf.math.reduce_max(diff_sum_32)
max_sum_16 = tf.math.reduce_max(diff_sum_16)
print(max_sum_32)
print(max_sum_16)
``` -->

  - mean error on float 32 : $7.8 \times 10^{-8}$
  - mean error on float 16 : $6.6 \times 10^{-4}$


and on convolutions of kernels $(5, 5, 96, 256)$ it gives
<!-- ```python
import tensorflow as tf
k = tf.random.uniform((5, 5, 96, 256), maxval=1/((5*5*96)), dtype=tf.float64)
i = tf.random.uniform((1, 50,50, 96), maxval=1, dtype=tf.float64)
out_64 = tf.nn.convolution(i,k)
out_32 = tf.cast(tf.nn.convolution(tf.cast(i, tf.float32), tf.cast(k, tf.float32)), tf.float64)
out_16 = tf.cast(tf.nn.convolution(tf.cast(i, tf.float16), tf.cast(k, tf.float16)), tf.float64)

out_64 = tf.reshape(out_64, 46*46*256)
out_32 = tf.reshape(out_32, 46*46*256)
out_16 = tf.reshape(out_16, 46*46*256)

diff_32 = tf.sort(tf.abs(out_64 - out_32))
diff_16 = tf.sort(tf.abs(out_64 - out_16))
max_32 = tf.math.reduce_max(diff_32)
max_16 = tf.math.reduce_max(diff_16)
print(max_32)
print(max_16)
``` -->

  - mean error on float 32 : $7.4 \times 10^{-7}$
  - mean error on float 16 : $1.3 \times 10^{-4}$

For the several implementation of quaternion multiplication, we can check these results experimentally

```python
import tensorflow as tf 
from upstride.type2.tf.keras import utils
for j in range(1, 100):
  i = [tf.random.uniform((1, 50,50, 96), maxval=1, dtype=tf.float32) for _ in range(4)]
  k = [tf.random.uniform((5, 5, 96, 256), maxval=1/(300), dtype=tf.float32) for _ in range(4)]
  conv_naive = lambda x, y : utils.quaternion_mult_naive(tf.nn.convolution, x,y)
  # conv1 = lambda x, y : utils.quaternion_mult1(tf.nn.convolution, x,y, j)
  conv2 = lambda x, y : utils.quaternion_mult2(tf.nn.convolution, x,y, j)
  out_32 = tf.concat(conv_naive(tf.cast(i, tf.float32), tf.cast(k, tf.float32)), axis=-1)
  out_16_n = tf.math.reduce_max(tf.sort(tf.abs(tf.cast(tf.concat(conv_naive(tf.cast(i, tf.float16), tf.cast(k, tf.float16)), axis=-1), tf.float32) - out_32))).numpy()
  # out_16_1 = tf.math.reduce_max(tf.sort(tf.abs(tf.cast(tf.concat(conv1(tf.cast(i, tf.float16), tf.cast(k, tf.float16)), axis=-1), tf.float32) - out_32))).numpy()
  out_16_2 = tf.math.reduce_max(tf.sort(tf.abs(tf.cast(tf.concat(conv2(tf.cast(i, tf.float16), tf.cast(k, tf.float16)), axis=-1), tf.float32) - out_32))).numpy()
  # print(j, out_16_1, out_16_2, out_16_n)
  print(j, out_16_2, out_16_n)


```

  * for naive conv, $\delta^{\mathbb{H}}_{conv} = 1.0 \times 10^{-3}$
  * for method 1, $\delta^{\mathbb{H}}_{conv} = 1.7 \times 10^{-3}$
  * for method 2, $\delta^{\mathbb{H}}_{conv} = 1.4 \times 10^{-3}$



## Other
  * Now the default docker image is running tensorflow 2.2.0.
