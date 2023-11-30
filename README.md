# NN-from-scratch
Development of a neural network using only basic math operations

### 1. [Introduction](#introduction)
### 2. [Math behind](#Math behind)
- #### 2.1. [Forward pass](#Forward pass)
- #### 2.2. [Contributing](#contributing)
- #### 2.3. [License](#license)

## Introduction
The project is a dense neural network with variable number of layers and layer sizes as well as its training and testing routines.
It was developed as a way to validate the knowledge aquired in the AI2 classes of THI University about how a dense neural network works in its lowest level.

## Mathbehind 

### 
$ŷ$ : Output activations vector (prediction)

$y$ : Label vector (how the output should be)

$x$ : Input vector

$\varphi(z)$ : Activation function

$L(y, ŷ)$ : Loss function

$w^{[l]}$ : Weight matrix for the layer l

$a^{[l]}$ : Activations matrix for the layer l

$b^{[l]}$ : Biases matrix for the layer l


### Forwardpass:

Activation function chosen: sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}} $


$z^{[l]} = w^{[l]} \cdot a^{[l-1]} + b^{[l]}$ 

$a^{[l]} = \varphi (z^{[l]})$

### Backpropagation:
Loss function chosen: RME $L(y, ŷ) = \frac{1}{2} \cdot(y - ŷ)^2$ 



### Optimization:

$ $



## Algorithm and function

Forward function


## Training process
This is the contributing section.

