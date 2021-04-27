#  Introduction

In this document we discuss the notions of Geometric Algebra (section Theory) and explain how GA is implemented in our UpStride engine (section Implementation).

- [Introduction](#introduction)
- [Theory](#thoery)
  - [Definitions](#definitions)
  - [Data representation](#data-representation)
  - [Linear layers](#linear-layers)
  - [Complex numbers](#complex-numbers)
  - [Quaternions](#quaternions)
  - [General case](#general-case)
  - [The Bias problem](#the-bias-problem)
  - [Non-linear layers](#non-linear-layers)
    - [BatchNormalization](#batchnormalization)
    - [Dropout](#dropout)
  - [Initialization:](#initialization)
  - [Data conversion](#data-conversion)
- [Implementation](#implementation)
  - [Using Upstride Engine](#using-upstride-engine)
    - [Factor parameter](#factor-parameter)
    - [Initialization](#initialization)
    - [TF2Upstride](#tf2upstride)
    - [Upstride2TF](#upstride2tf)
- [References](#references)

# Theory
## Definitions

Before we proceed lets look at some of the definitions and notations we will be using in this document.

* _blade_ - A blade is a generalization of the concept of scalars and vectors. Specifically, a $k$-blade is any object that can be expressed as the exterior product (or wedge $\wedge$ product) of $k$ vectors, and is of grade $k$.
* _Multivector_ - A multivector is a linear combination of k-blades.
* $\mathbb{G} \circ \mathbb{M}$ - GA represented as real values in the matrix form. This is crucial as numerical computation frameworks like Tensorflow do not support GA yet.
* $\mathbb{M} \circ \mathbb{G}$ - Matrices of GA. This representation has difficult to work with the frameworks like TensorFlow as we don't have a data type that can natively represent GA.
* $x$ - Inputs to the Neural Network layer
* $y$ - Outputs to the Neural Network layer
* $W$ - Weights to the Neural Network layer
* $\sum$ - Summation
* $\beta_i$ - Represents the ith-blade
* $\mathbb{R^3}$ - a vector space of dimension 3 over the field $\mathbb R$ of real numbers.
* $\wedge$ - exterior product or wedge product

The Geometric Algebra (GA) represenatation is implemented in python as $\mathbb{G} \circ \mathbb{M}$, or geometrical algebra over matrices. Code is written in TensorFlow using Keras high-level API and supports Python 3.6 or higher.

The goal of th1182is document is to provide all the mathematical explanations and algorithm details to understand the code.

## Data representation

We stack the _blades_ on the batch dimension of the tensor. So, for instance, if we are working in a GA with 4 blades, a image tensor will have the shape :

(4 * BS, C, H, W)

with

BS: the batch size

C : number of channels

H : height

W : width

Its important to note:

- Blades are NOT interleaved with regards to batch. It means that the inner data representation is of shape (4, BS, C, H, W) and not (BS, 4, C, H, W). For example, to get the first full feature map, you need to type tensor[::BS] and to get all the real values, you need to type tensor[:BS]

- When performing the conversion between real and upstride, the only change the user will notice is this increased batch size. If the neural network is using the information of C, H, W or H, W, C, then user doesn’t need to update his model definition when using upstride.

- Although the above example follows channels first convention, the UpStride engine supports channels last data format as well.


## Linear layers

This section describes how to implement any linear layer from any GA in TensorFlow, for instance:

- Conv2D
- Dense
- DepthwiseConv2D
- Conv2DTranspose (experimental)

The idea is to implement a very generic version of a linear layer, valid for any GA and any linear operation. Then all the specific implementations will benefit from generic implementation.

Note: Conv2DTranspose layer is experimental and it hasn't been throughly validated.

Note: `SeparableConv2D` is an exception. It is computed by going through 2 linear functions, but moving these 2 linear function to hypercomplex is not the same as moving the combination of the function to hypercomplex. Currently not supported by the UpStride engine for any GA.

Let's go over an example on how generic linear layer work. In the following two sections we describe two specific GAs, that is complex numbers and quaternions.

## Complex numbers

Let’s define:

$x = x_R + ix_I$, the complex input of a linear layer

$y = y_R + iy_I$, the complex output of the same layer

$W = W_R + iW_I$, the kernel of the layer

Computing linear layer means to compute the product : $y = xW$

So to compute $y$, we first need to compute all the cross-product between the components of $x$ and the components of $W$. This can be done in a single call to the Tensorflow API.

Indeed, as we saw in the data representation section, for tensorflow $x$ is a single tensor which is the concatenation of $x_R$ and $x_I$ on axis 0 (the BS axis). Now we need to concatenate the component of the kernel along the output channel axis.

So, for instance a linear layer will have:

- $x_R$ and $x_I$ are tensors of shape (BS, C)

- $x$ is a tensor of shape (2 * BS, C)

- $W_R$ and $W_I$  are tensors of shape (C, C$^\prime$)

- $W$ is a tensor of shape (C, 2 * C$^\prime$)

To compute the linear product $y = xW$, the output will be a tensor of shape (2 * BS, 2* C$^\prime$) equal to:

$\begin{bmatrix}
x_R W_R , x_R W_I \\
x_I W_R , x_I W_I
\end{bmatrix}$

We need to split the matrix and aggregate the component results:

$y_R = x_R W_R - x_I W_I$

$y_I = x_R W_I + x_I W_R$

## Quaternions

Let look at example of computing the linear product for quaterions.

Given two quaternions $u = u_1 + u_2i + u_3j + u_4k$ and $v= v_1 + v_2i + v_3j + v_4k$, the naive way to compute the product $c$ is :

$\begin{array}{rl}
c_1 = &  u_1 v_1 -  u_2 v_2 -  u_3 v_3 -  u_4 v_4 \\
c_2 = &  u_1 v_2 +  u_2 v_1 +  u_3 v_4 -  u_4 v_3 \\
c_3 = &  u_1 v_3 +  u_3 v_1 +  u_4 v_2 -  u_2 v_4 \\
c_4 = &  u_1 v_4 +  u_4 v_1 +  u_2 v_3 -  u_3 v_2
\end{array}$

This computation includes 16 multiplications and 12 additions.Due to the isomorphism between $\mathbb{M} \circ \mathbb{G}$ and $\mathbb{G} \circ \mathbb{M}$, this corresponds to 16 calls to the TensorFlow linear layer of choice.

## General case

Let’s work with a generic geometrical algebra defined by a set of blades $\beta_i, i \in [0, n]$.

A blade is a generalization of the concept of scalars and vectors to include multi-vectors. Specifically, a $k$-blade is any object that can be expressed as the exterior product (or wedge $\wedge$ product) of $k$ vectors, and is of grade $k$. For example, the complex number $x = x_R + ix_I$ can be expressed in terms of blades as $x = x_R\beta_0+ x_I\beta_1$.

In general, we can then write our 3 elements $x$, $y$ and $W$ as:

- $x = \sum_ix_i\beta_i$
- $y = \sum_iy_i\beta_i$
- $W = \sum_iW_i\beta_i$

Then,

$y = \sum_i\sum_jx_iW_j\beta_i\beta_j$

the product $\beta_i\beta_j$ is defined by the structure of the GA and can be expressed as:
$\beta_i\beta_j = s\beta_k,(s,k) \in \lbrace -1, 0, 1\rbrace \times [0, n]$

Note that the set of blades $\beta_i,i \in [0,n]$ does not correspond necessarily to an orthonormal set . For example, given the orthonormal set $\lbrace e_1, e_2, e_3 \rbrace$ in the space $\mathbb{R}^3$, a valid basis for a GA on the same space is $\lbrace 1, e_1, e_2, e_3, e_1 \wedge e_2, e_2 \wedge e_3, e_3 \wedge e_1, e_1 \wedge e_2 \wedge e_3 \rbrace$. We can express this basis in terms of blades as $\beta_i, i \in [0, 7]$ where  is $\beta_0$ a 0-blade, $\beta_1, \beta_2, \beta_3$ are 1-blades, $\beta_4, \beta_5, \beta_6$ are 2-blades and $\beta_7$ is a 3-blade.

 Computing the product with a singe operation is done in the same way as for the case of complex numbers, that is by concatenating weights $W_i$'s.

Then we need a function that computes the product of two blades:i.e. a function that takes (i, j) as input and returns (k, s)

```python
def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
```

Now, we have everything to code the GenericLinear operation. Note that we do not need to know which linear TensorFlow operation will be used. We can pass this information as an argument.

## The Bias problem

In deep learning, we often add a bias term after a linear operation. In Keras, this bias is handled by the linear layer itself, which is a problem here.

Let's take the example of complex number. If the bias is in the TensorFlow layer the computation will be:

$y_R = x_RW_R - x_IW_I + b_R - b_I$
$y_I = x_IW_R + x_RW_I + b_R + b_I$

with $b_R, b_I$ are the bias terms.

This formulation has two issues:

1.  we perform two more operations (the bias terms) than needed.
2.  If we only work with one component of the multivector (e.g: $y_R$ as the final output of the softmax layer) then we have two variables ($b_R$ and $b_I$) and one constraint. This kind of situation can hurt performance.

One solution to prevent this is to detect when the user wants the bias term and handle it by following the steps:

- The user sends the parameters to create a linear layer with bias
- We save the bias parameters and do not forward these to Keras
- Keras creates the linear layer without the bias
- We perform the GA multiplication
- We add at bias at the end with the user's parameters

## Non-linear layers

Examples of non-linear layers are:

- MaxPooling2D
- GlobalAveragePooling2D
- Reshape
- BatchNormalization
- Activation
- Flatten
- ZeroPadding2D
- Add
- Concatenate

For most of the non-linear layer, the equation is simply:

$y = \sum_if_i(x_i)\beta_i$

where,

$y$ is the output

$x_i$ is the input

$\beta_i$ is the blade

$f_i$ is the non-linear function

With the way we encode hypercomplex tensor, the non-linear hypercomplex layer is the same as the real layer. The sum in the previous equation is handle naturally as all blades are stack along the batch size. However, there are some exceptions, such as BatchNormalization and Droput. We describe those in the following two sections.

### BatchNormalization

For batchnormalization to work, the blades shouldn’t be stack along the first axis but the channel axis. Stacking along the channel axis will allow the normalization to be computed independently for every blades.

Also, note that the current (BatchNormalization) implementation works on the several components of the multivector in a correlated way. If you want more clever mechanism you should go for BatchNormalizationC (Complex) or BatchNormalizationH (Quaternion) where the real and imaginary parts are decorrelated.

**BatchNormalizationC**: Trabelsi et al [1]

* Ensures the equal variance for both real and imaginary parts, unlike applying real valued BatchNormalization.
* Removes correlation between real and imaginary parts.

Its recommended to read Section 3.5 in the [Deep Complex Networks](https://arxiv.org/pdf/1705.09792.pdf#subsection3.5) paper for further details.

**BatchNormalizationH**: Gaudet et al [2]

* Similar to Complex BatchNormalization. The Quaternion BatchNormalization uses the same idea to ensure all 4 components to have equal variance.
* Removes correlattion between real and each imaginary parts.


It's recommended to read section 3.4 in the [Deep Quaternion Networks](https://arxiv.org/pdf/1712.04604.pdf#subsection.3.4) paper for further details.

### Dropout

Dropout is also an exception because we want to apply dropout on all components of a hypercomplex number at the same time. This can't be done with a single Dropout.

The solution is to define N Dropout operation with the same random seed to synchronize them. Then at every iteration, we split the input tensor to N, compute the N dropout and concatenate the output.

## Initialization:

UpStride engine supports all the standard weight initialization techniques thats supported by TensorFlow/Keras API.

The weight initialization techniques for Complex (Trabelsi el al [1]) and Quarternion (Gaudet el al [2]) are supported. We have used them in the past, but they are not extensively validated and are considered experimental.

## Data conversion

Two operations are defined for converting data between TensorFlow and UpStride : TF2Upstride and Upstride2TF.

These 2 operations support several strategies depending on the Upstride type we’re using.

For **TF2Upstride**:

- `default` or `basic`: Tensorflow Tensor is placed in the real part and imaginary parts are initialized with zeros.
- `learned`: A resnet block (BN -> RELU -> CONV -> BN -> RELU -> CONV) is used to learn the imaginary parts. (Trabelsi et al [1])

For **Upstride2TF**:

- `basic`: outputs a tensor that keeps only the real values of the tensor.
- `concat`: generates a vector by concatinating the imaginary components on the final dimension.
- `max_pool`: ouputs a tensor that takes the maximum values across the real and imaginary parts.
- `avg_pool`: ouputs a tensor that takes average across the real and imaginary parts.


# Implementation

UpStride integrates seamlessly with TensorFlow 2.4. It is assumed the user is experienced with development based on TensorFlow/Keras.

## Using Upstride Engine

This is a simple neural network that uses UpStride layers:

```python
import tensorflow as tf
from upstride.typetf.keras import layers

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = layers.TF2Upstride()(inputs)
x = layers.Conv2D(32, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(100)(x)
x = layers.Upstride2TF()(x)

model = tf.keras.Model(inputs=[inputs], outputs=[x])
```

UpStride engine provides layers that transforms real valued tensor into the following representations.

- type1 - Complex
- type2 - Quaternion

Note: we also have a *type0* which is the same as using tensorflow tensors with real values.

Every module encapsulate a API similar to Keras.

To use it,
- start by importing the layers package from the upstride type you want to use.
- Start by defining a Input to the network.
- convert Tensorflow tensor to Upstride type by calling `layers.TF2Upstride`
- build the neural network the same way you do with Keras.
- At the end of the neural network call `layers.Upstride2TF` to convert from upstride  to TensorFlow tensor.

For training and inference, all TensorFlow tools can be used (distribute strategies, mixed-precision training...)

UpStride's engine is divided in three main modules `type0`, `type1` and `type2`. Every module has the same type of layers and functions. They can be imported in the following way:

```python
# type 0
from upstride.type0.tf.keras import layers
# type 1
from upstride.typetf.keras import layers
# type 2
from upstride.typetf.keras import layers
```

When a network is built with `type0`, it is equivalent to the real valued network or just using tensorflow layers without upstride engine.

In the following three sections we describe some specific features implemented in our framework, such as the `factor` parameter, and conversion strategies: `Upstride2TF` and `TF2Upstride`.
### Factor parameter

This is a simple neural network that uses UpStride layers (type 2) with `factor` parameter:

```python
import tensorflow as tf
from upstride.typetf.keras import layers

factor = 4

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = layers.TF2Upstride()(inputs)
x = layers.Conv2D(32 // factor, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.Conv2D(64 // factor, (3, 3))(x)
x = layers.Activation('relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(100 // factor)(x)
x = layers.Upstride2TF()(x)

model = tf.keras.Model(inputs=[inputs], outputs=[x])
```

The `factor` parameter is used to linearly scale the number of feature maps in the linear layers. Higher factor value results in less number of feature maps and vice versa. By using a factor parameter compromise between overall accuracy performance and total number of free parameters of a model. The factor can be applied to all the linear layers except the final logits layer.

For example,

```python
x = layers.Conv2D(32 // factor, (3, 3))(x)
```

Here, `factor = 2` reduces the number of channels to 16, `factor = 4` reduces the number of channels to 8.

The factor can be used as a hyper parameter to reduce the network width for the linear layers which would impact the final performance. The factor can be any `int` value. Ensure the value is not too small enough to hinder the learning capability of the network.

In the above example, the ouput channels for the `Conv2D` is 3 If `factor = 16` then resulting output will be `1`. The network will struggle to extract features with just 1 output channel.

The `factor` scales the number of channels for all the linear layers when applied. This helps in controlling the capacity of the overall network.

However, due to the way the UpStride engine is implemented, the vanilla approach (without using the `factor` i.e. when `factor` = 1) results in a model that contains more free parameters than its pure TensorFlow counterpart.

### Initialization

Weight initialization are done similar to TensorFlow/Keras. We utilize the `kernel_initializer` parameter in the linear layers.

For example,

```python
from upstride.typetf.keras import layers
# ...
# ...
x = layers.Conv2D(32 // factor, (3, 3), kernel_initializer='glorot')(x)
```

Note: There are specific functionality for each `type0`, `type1` and `type2` upstride types.

Refer `InitializersFactory` in the folder `upstride/initializers.py` for further information.

### TF2Upstride

As we have seen from [data conversion](#data-conversion) section there are 2 strategies available to convert TensorFlow tensors to Upstride tensors.

Each strategy can be passed as parameters when invoking `TF2Upstride`

```python
x = layers.TF2Upstride(strategy="default")(inputs)
```
```python
x = layers.TF2Upstride(strategy="learned")(inputs)
```

### Upstride2TF

As we have seen from [data conversion](#data-conversion) section there are 4 strategies available to convert UpStride tensors to TensorFlow tensors.

Each strategy can be passed as parameters when invoking `Upstride2TF`

e.g:
```python
x = layers.Upstride2TF(strategy="default")(x)
```

```python
x = layers.Upstride2TF(strategy="concat")(x)
```

```python
x = layers.Upstride2TF(strategy="max_pool")(x)
```

```python
x = layers.Upstride2TF(strategy="avg_pool")(x)
```
# References

1. Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal. “Deep Complex Networks”. In Internation Conference on Learning Representations (ICLR), 2018

2. Gaudet, Chase J., and Anthony S. Maida. "Deep quaternion networks." 2018 International Joint Conference on Neural Networks (IJCNN). IEEE, 2018.