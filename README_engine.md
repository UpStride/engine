# 1. Introduction

In this document we discuss the notions of Geometric Algebra (GA) (section 1.1.Theory) and explain how GA is implemented in our python engine (section 1.2.Implementation).

- [1. Introduction](#1-introduction)
  - [1.1. Theory](#11-theory)
    - [1.1.1. Definitions](#111-definitions)
    - [1.1.2. Data representation](#112-data-representation)
    - [1.1.3. Linear layers](#113-linear-layers)
    - [1.1.4. Complex numbers](#114-complex-numbers)
    - [1.1.5. Quaternions](#115-quaternions)
    - [1.1.6. General case](#116-general-case)
    - [1.1.7. The Bias problem](#117-the-bias-problem)
    - [1.1.8. Non-linear layers](#118-non-linear-layers)
      - [1.1.8.1. BatchNormalization](#1181-batchnormalization)
      - [1.1.8.2. Dropout](#1182-dropout)
    - [1.1.9. Data conversion](#119-data-conversion)
  - [1.2. Implementation](#12-implementation)
    - [1.2.1. Getting Upstride Engine](#121-getting-upstride-engine)
    - [1.2.2. Using Upstride Engine](#122-using-upstride-engine)
      - [1.2.2.1. Factor parameter](#1221-factor-parameter)
      - [1.2.2.2. TF2Upstride](#1222-tf2upstride)
      - [1.2.2.3. Upstride2TF](#1223-upstride2tf)
  - [1.3. References](#13-references)
  
## 1.1. Theory
### 1.1.1. Definitions 

Before we proceed lets look at some of the definitions and notations we will be using in this document. 

* _Blade_ - A blade is a generalization of the concept of scalars and vectors. Specifically, a $k$-blade is any object that can be expressed as the exterior product (or wedge $∧$ product) of $k$ vectors, and is of grade $k$.
* _Multivector_ - A multivector is a linear combination of k-blades.
* $\mathbb{G} \circ \mathbb{M}$ (Geometric Algebra over Matrices) - GA represented as real values in the matrix form. This is crucial as numerical computation frameworks like Tensorflow do not support GA yet.
* $\mathbb{M} \circ \mathbb{G}$ (Matrices over Geometric Algebra) - This representation has a difficult integration with frameworks like TensorFlow as we don't have a data type that can natively represent GA. 
* $x$ - Inputs to the Neural Network layer
* $y$ - Outputs to the Neural Network layer
* $W$ - Weights to the Neural Network layer
* $\sum$ - Summation 
* $\beta_i$ - Represents the i<sup>th</sup>-blade 
* $\mathbb{R^3}$ - a vector space of dimension 3 over the field $\mathbb R$ of real numbers.
* $\wedge$ - exterior product or wedge product

The Geometric Algebra representation is implemented in python as $\mathbb{G} \circ \mathbb{M}$. Code is written in TensorFlow 2.4.1 using Keras high-level API and supports Python 3.6 or higher. 

The goal of this document is to provide all the mathematical explanations and algorithm details to understand the code.
### 1.1.2. Data representation

We stack the _blades_ on the batch dimension of the tensor. So, for instance, if we are working in a GA with 4 blades, a image tensor will have the shape $(4 \times BS, C, H, W)$, with:

$BS$: the batch size
$C$: number of channels
$H$: height
$W$: width

Its important to note:

- Blades are NOT interleaved with regards to batch. It means that the underlying data representation is of shape $(4, BS, C, H, W)$ and not $(BS, 4, C, H, W)$. For example, to get the first full feature map, you need to type `my_tensor[::4]` and to get all the real values, you need to type `my_tensor[:BS]`.

- When performing the conversion between real and upstride, the only change the user will notice is this increased batch size.

- Although the above example follows `channels_first` data format convention, the python engine supports `channels_last` as well.


### 1.1.3. Linear layers

This section describes how to implement any linear layer from any GA in TensorFlow, for instance:

- `Dense`
- `Conv2D`
- `DepthwiseConv2D`
- `Conv2DTranspose` (experimental)

The idea is to implement a very generic version of a linear layer, valid for any GA and any linear operation. Then all the specific implementations will benefit from generic implementation.

Note: Conv2DTranspose layer is experimental and we have not thoroughly validated. 

Note: `SeparableConv2D` is an exception. It is computed by going through 2 linear functions, but moving these 2 linear functions to hypercomplex is not the same as moving the combination of the function to hypercomplex. Currently not supported by the python engine for any GA.

Let's go over an example on how generic linear layer work. In the following two sections we describe two specific GAs, that is complex numbers and quaternions.

### 1.1.4. Complex numbers

Let’s define:

$x = x_R + ix_I$, the complex input of a linear layer

$y = y_R + iy_I$, the complex output of the same layer

$W = W_R + iW_I$, the kernel of the layer

Computing linear layer means to compute the product: $y = xW$

So to compute $y$, we first need to compute all the cross-product between the components of $x$ and the components of $W$. This can be done in a single call to the TensorFlow API.

Indeed, as we saw in the data representation section, for TensorFlow $x$ is a single tensor which is the concatenation of $x_R$ and $x_I$ on axis 0 (the BS axis). Now we need to concatenate the component of the kernel along the output channel axis.

So, for instance a linear layer will have: 

- $x_R$ and $x_I$ are tensors of shape $(BS, C)$

- $x$ is a tensor of shape $(2 \times BS, C)$

- $W_R$ and $W_I$  are tensors of shape $(C, C^\prime)$

- $W$ is a tensor of shape $(C, 2 \times C^\prime)$

To compute the linear product $y = xW$, the output will be a tensor of shape $(2 \times BS, 2\times C^\prime)$ equal to:

$\begin{bmatrix}
x_RW_R , x_RW_I \\
x_IW_R , x_IW_I
\end{bmatrix}$

so we get $y = \sum_i\sum_jx_iW_j$

### 1.1.5. Quaternions

Let's look at example of computing the linear product for quaternions.

Given two quaternions $u = u_1 + u_2i + u_3j + u_4k$ and $v= v_1 + v_2i + v_3j + v_4k$, the naive way to compute the product $c$ is :

$\begin{array}{rl}
c_1 = &  u_1 v_1 -  u_2 v_2 -  u_3 v_3 -  u_4 v_4 \\
c_2 = &  u_1 v_2 +  u_2 v_1 +  u_3 v_4 -  u_4 v_3 \\
c_3 = &  u_1 v_3 +  u_3 v_1 +  u_4 v_2 -  u_2 v_4 \\
c_4 = &  u_1 v_4 +  u_4 v_1 +  u_2 v_3 -  u_3 v_2 
\end{array}$

This computation includes 16 multiplications and 12 additions. Due to the isomorphism between $ \mathbb{M} \circ \mathbb{G}$ and $\mathbb{G} \circ \mathbb{M}$, this corresponds to 16 calls to the TensorFlow linear layer of choice.

### 1.1.6. General case

Let’s work with a generic geometrical algebra defined by a set of blades $\beta_i, i \in [0, n]$.

A blade is a generalization of the concept of scalars and vectors to include multivectors. Specifically, a $k$-blade is any object that can be expressed as the exterior product (or wedge $\wedge$ product) of $k$ vectors, and is of grade $k$. For example, the complex number $x = x_R + ix_I$ can be expressed in terms of blades as $x = x_R\beta_0+ x_I\beta_1$.

In general, we can then write our 3 quantities $x$, $y$ and $W$ as:  

- $x = \sum_ix_i\beta_i$
- $y = \sum_iy_i\beta_i$
- $W = \sum_iW_i\beta_i$

Then,

$y = \sum_i\sum_jx_iW_j\beta_i\beta_j$

the product $\beta_i\beta_j$ is defined by the structure of the GA and can be expressed as:  
$\beta_i\beta_j = s\beta_k,(s,k) \in \lbrace -1, 0, 1\rbrace \times [0, n]$

Note that the set of blades $\beta_i,i \in [0,n]$ does not correspond necessarily to an orthonormal set . For example, given the orthonormal set $\lbrace e_1, e_2, e_3 \rbrace$ in the space $\mathbb{R}^3$, a valid basis for a GA on the same space is $\lbrace 1, e_1, e_2, e_3, e_1 \wedge e_2, e_2 \wedge e_3, e_3 \wedge e_1, e_1 \wedge e_2 \wedge e_3 \rbrace$. We can express this basis in terms of blades as $\beta_i, i \in [0, 7]$ where  is $\beta_0$ a 0-blade, $\beta_1, \beta_2, \beta_3$ are 1-blades, $\beta_4, \beta_5, \beta_6$ are 2-blades and $\beta_7$ is a 3-blade.

 Computing the product with a single operation is done in the same way as for the case of complex numbers, that is by concatenating weights $W_i$'s.

Then we need a function that computes the product of two blades, i.e. a function that takes as input the indexes $(i, j)$ of the two blades to be multiplied and returns the index $k$ and the sign $s$ of such product.

```python
def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
```

Now, we have everything to code the GenericLinear operation. Note that we do not need to know which linear TensorFlow operation will be used. We can pass this information as an argument.

### 1.1.7. The Bias problem

In deep learning, we often add a bias term after a linear operation. In Keras, this bias is handled by the linear layer itself, which is a problem here.

Let's take the example of complex number. If the bias is in the TensorFlow layer the computation will be:

$y_R = x_RW_R - x_IW_I + b_R - b_I$
$y_I = x_IW_R + x_RW_I + b_R + b_I$

with $b_R, b_I$ are the bias terms.

This formulation has two issues:

1.  we perform two more operations (the bias terms) than needed.
2.  If we worked with a single blade of the multivector (e.g: $y_R$ as the final output of the softmax layer) then we would have two variables ($b_R$ and $b_I$) and only one constraint. This kind of situation can hurt performance.

One solution to prevent this is to detect when the user applies the bias term and handle it by following the steps:

1. The user sends the parameters to create a linear layer with bias 
2. We save the bias parameters and do not forward these to Keras
3. Keras creates the linear layer without the bias
4. We perform the GA multiplication
5. We add at bias at the end with the user's parameters

### 1.1.8. Non-linear layers

Examples of non-linear layers are:

- `MaxPooling2D`
- `GlobalAveragePooling2D`
- `Reshape`
- `BatchNormalization`
- `Activation`
- `Flatten`
- `ZeroPadding2D`
- `Add`
- `Concatenate`

The idea here is again to define a class `GenericNonLinear` that all the non-linear layers inherit. 

For most of the non-linear layer, the equation is simply:

$y = \sum_if_i(x_i)\beta_i$

where, 
- $y$ is the output
- $x_i$ is the input
- $\beta_i$ is the blade
- $f_i$ is the non-linear function

With the way we encode hypercomplex tensor, the non-linear hypercomplex layer is the same as the real layer. The sum in the previous equation is handled naturally as all blades are stacked along the batch axis. However, there are some exceptions, such as BatchNormalization and Dropout. We describe those in the following two sections.

#### 1.1.8.1. BatchNormalization

For batchnormalization to work, the blades should not be stacked along the first axis so that the normalization is not computed as if the blades belonged to the batch axis. We decided to stack the blades on the channel axis.

Also, note that the current (BatchNormalization) implementation works on the several components of the multivector in a correlated way. If you want more clever mechanism you should go for BatchNormalizationC (Complex) or BatchNormalizationH (Quaternion) where the real and imaginary parts are decorrelated. 

**BatchNormalizationC**: Trabelsi et al [1]

* Ensures the equal variance for both real and imaginary parts, unlike applying real valued BatchNormalization.
* Removes correlation between real and imaginary parts.

It's recommended to read Section 3.5 in the [Deep Complex Networks](https://arxiv.org/pdf/1705.09792.pdf#subsection3.5) paper for further details.

**BatchNormalizationH**: Gaudet et al [2]

* Similar to Complex BatchNormalization. The Quaternion BatchNormalization uses the same idea to ensure all 4 components to have equal variance.
* Removes correlation between real and each imaginary parts.
 

It's recommended to read section 3.4 in the [Deep Quaternion Networks](https://arxiv.org/pdf/1712.04604.pdf#subsection.3.4) paper for further details.

#### 1.1.8.2. Dropout

Dropout is also an exception because we want to apply dropout on all blades of a hypercomplex number at the same time. This can't be done with a single Dropout.

Let $N$ be the number of blades. The solution is to define $N$ Dropout operations with the same random seed to synchronize them. Then at every iteration, we split the input tensor to $N$, compute the N dropout and concatenate the output.

### 1.1.9. Data conversion

Two operations are defined for converting data between TensorFlow and Upstride: TF2Upstride and Upstride2TF.

These 2 operations support several strategies depending on the Upstride type we’re using.

For **TF2Upstride**: 

- `default` or `basic`: The TensorFlow Tensor is placed in the real component and all the other blades are initialized with zeros.
- `learned`: A resnet block (BN -> ReLU -> CONV -> BN -> ReLU -> CONV) is used to learn the imaginary parts. (Trabelsi et al [1])

For **Upstride2TF**:

- `basic`: outputs a tensor that keeps only the real component of the tensor. 
- `concat`: generates a vector by concatenating the imaginary components on the channel dimension.
- `max_pool`: (experimental) outputs a tensor that takes the maximum values across the real and imaginary parts.
- `avg_pool`: (experimental) outputs a tensor that takes the average across the real and imaginary parts.



## 1.2. Implementation

Upstride integrates seamlessly with TensorFlow 2.4. It is assumed that the user is experienced with development based on TensorFlow/Keras.

We offer two ways to use Upstride, either on the cloud or on premises:

1. _Upstride Cloud_:
  - Make training and inference requests using the Upstride REST API
  - Expose trained models to the internet by creating unique endpoints
  - Command-line interface (CLI) and simple web interface
2. _Upstride Enterprise_:
  - Docker is shipped to the client
  - It can be installed in a local server/machine (requires the purchase of a license)
  - Access to full-blown functionalities of the Upstride Python API.

Here is how Upstride integrates within a training/inference AI pipeline:

![](images/api_overall.png)

The outer box represents the Upstride Docker container. The user runs the docker, loads the datasets and scripts, and uses the Upstride API to build neural networks.

### 1.2.1. Getting Upstride Engine

Upstride's products are packaged into [docker](https://docs.docker.com) containers images. The images are available in private docker registry listed below:

`registry.us.upstride.io` --> for clients in the US

`registry.upstride.io` --> for clients in the Europe


It is mandatory to use the docker [cli](https://docs.docker.com/engine/reference/commandline/cli/) to interact with the private registries. Don't try to download the content of the registry with the web browser. You will be automatically redirected to this document.

You only need two pieces of information to download containers images in the registry: 

1. Login credentials i.e. username and password 
2. The exact docker image's name

Those information cannot be guessed. There are provided during Upstride's onboarding or products delivery. Please contact us to have more information [@upstride.](https://ww2.upstride.io/contact-us-page/)

When you receive those information, use standard docker commands to login and pull docker images: 

```
docker login -u username -p password registry.us.upstride.io
docker pull registry.us.upstride.io/image_repo/image_name:image_tag
```

### 1.2.2. Using Upstride Engine

This is a simple neural network that uses Upstride layers:

```python
import tensorflow as tf
from upstride.type2.tf.keras import layers

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

Upstride engine provides layers that transforms real valued tensor into the following representations.

- type1 - Complex 
- type2 - Quaternion

Each module encapsulates an API similar to Keras.

To use it, 
- import the layers package from the upstride type you want to use, e.g.`import upstride.type2.tf.keras.layers as layers`.
- define an Input to the network. 
- convert TensorFlow tensor to Upstride type by calling `layers.TF2Upstride`
- build the neural network the same way you do with Keras. 
- at the end of the neural network call `layers.Upstride2TF` to convert from upstride  to TensorFlow tensor.

For training and inference, all TensorFlow tools can be used (distribute strategies, mixed-precision training...)

Upstride's engine is divided in two main modules `type1`, `type2`. All the modules have the same type of layers and functions. They can be imported in the following way:

```python
# type 1
from upstride.type1.tf.keras import layers
# type 2
from upstride.type2.tf.keras import layers
```

In the following three sections we describe some specific features implemented in our framework, such as the `factor` parameter, and conversion strategies: `Upstride2TF` and `TF2Upstride`.
#### 1.2.2.1. Factor parameter

This is a simple neural network that uses Upstride layers (type 2) with `factor` parameter:

```python
import tensorflow as tf
from upstride.type2.tf.keras import layers

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

The `factor` parameter is used to linearly scale the number of feature maps in the linear layers. Higher factor value results in less number of feature maps and vice versa. The `factor` parameter allows to easily tradeoff between the overall accuracy performance and the total number of free parameters of a model. The factor can be applied to all the linear layers except the final logits layer. 

For example, 

```python 
x = layers.Conv2D(32 // factor, (3, 3))(x)
```

Here, `factor = 2` reduces the number of channels to 16, `factor = 4` reduces the number of channels to 8.

The factor value is used in conjunction with number of blades. For instance for type 2 (Quaternion), we use factor 4. In general, for an algebra of $k$-blades, we use factor equals to $k$ to compensate for the time complexity.

Note: There are no strict rules to use the above convention. The factor can be used as a hyper parameter to trim down the network width for the linear layers which would impact the final performance. The factor can be any integer value. Ensure the value is not excessively small, hindering the learning capability of the network. 

In the above example, the output channels for the `Conv2D` is 32. If `factor = 16` then resulting output will be `1`. The network will struggle to extract features with just 1 output channel.  

The `factor` scales the number of channels for all the linear layers when applied. This helps in controlling the capacity of the overall network.

However, due to the way the python engine is implemented, the vanilla approach (without using the `factor` i.e. when `factor` = 1) results in a model that contains more free parameters than its pure TensorFlow counterpart. 

#### 1.2.2.2. TF2Upstride

As we have seen from [data conversion](#219-data-conversion) section there are 2 strategies available to convert TensorFlow tensors to Upstride tensors.

Each strategy can be passed as parameters when invoking `TF2Upstride`

```python
x = layers.TF2Upstride(strategy="default")(inputs)
```
```python
x = layers.TF2Upstride(strategy="learned")(inputs)
```

#### 1.2.2.3. Upstride2TF

As we have seen from [data conversion](#219-data-conversion) section there are 4 strategies available to convert Upstride tensors to TensorFlow tensors.

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
## 1.3. References

1. Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal. “Deep Complex Networks”. In Internation Conference on Learning Representations (ICLR), 2018

2. Gaudet, Chase J., and Anthony S. Maida. "Deep quaternion networks." 2018 International Joint Conference on Neural Networks (IJCNN). IEEE, 2018.