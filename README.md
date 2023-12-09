# NN-from-scratch
Development of a neural network using only basic math operations

### 1. [Introduction](#introduction)
### 2. [Mathbehind](#Mathbehind)

- #### 2.1. [Forwardpass](#Forwardpass)
- #### 2.2. [Contributing](#contributing)
- #### 2.3. [License](#license)

## Introduction
The project is a dense neural network with variable number of layers and layer sizes as well as its training and testing routines.
It was developed as a way to validate the knowledge aquired in the AI2 classes of THI University about how a dense neural network works in its lowest level.

## Math behind 

$ŷ$ : Output activations vector (prediction)

$y$ : Label vector (how the output should be)

$x$ : Input vector

$\varphi(z)$ : Activation function

$L(y, ŷ)$ : Loss function

$w^{[l]}$ : Weight matrix for the layer l

$a^{[l]}$ : Activations matrix for the layer l

$b^{[l]}$ : Biases matrix for the layer l


<br>

<p align="center">
    <img src="images/net_diagram.png" width="700" />

</p>

### Forward pass:
It is the process o making the prediction, calculating the activation layer by layer until the last one, the output.
<p align="center">
    <img src="images/forward.png" width="1000" />

</p>

Activation function chosen: sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}} $


$z^{[l]} = w^{[l]} \cdot a^{[l-1]} + b^{[l]}$ 

$a^{[l]} = \varphi (z^{[l]})$

### Backpropagation:
It is the process of calculating the gradients of the Loss function relative to the weights and biases, done by calculating the derivatives of each function and applying the chaing rule.

<p align="center">
    <img src="images/backprop_derivs.png" width="1000" />

</p>

$\frac{{\ dL}}{{da_i^{[i]}}} = $

Loss function chosen: RME $L(y, ŷ) = \frac{1}{2} \cdot(y - ŷ)^2$ 

$\frac{{\ dL}}{{da^{[l]}}} = a^{[l]} - y$

$\frac{da^{[l]}}{{dz^{[l]}}} = a^{[l]} \cdot (1 - a^{[l]})$

$\frac{dz^{[l]}}{{da^{[l-1]}}} = w^{[l]}$



$\frac{dz^{[l]}}{{dw^{[l]}}} = a^{[l-1]}$

$\frac{dz^{[l]}}{{db^{[l]}}} = 1$



### Optimization:
Is the process of updating the weights

Stochastic gradient descent:

$w_{new} = w - \alpha \cdot \frac{dz^{[l]}}{{dw^{[l]}}}$

$b_{new} = b - \alpha \cdot \frac{dz^{[l]}}{{db^{[l]}}}$

Adam:

$\frac{dz^{[l]}}{dw^{[l]}}$

$w_{new} = w - \alpha \cdot \frac{\beta_1 \cdot m_{k-1} + (1-\beta_1) \cdot \frac{dL}{dw}}{\epsilon + \sqrt{\beta_2 \cdot m_{k-1} + (1-\beta_2) \cdot (\frac{dL}{dw})^2}}$

$b_{new} = b - \alpha \cdot \frac{dz^{[l]}}{{db^{[l]}}}$


## Algorithm and function

Forward function


## Training process
This is the contributing section.

