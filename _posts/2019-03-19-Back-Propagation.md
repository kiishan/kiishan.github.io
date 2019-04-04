---
layout: post
title: Understanding Back Propagation
date: 2019-03-19T00:00:00.000Z
published: true
categories: Deep Learning
read_time: true
---

# Back propagation

Backpropagation, short for "backward propagation of errors," is an algorithm for supervised learning of artificial neural networks using gradient descent. Given an artificial neural network and an error function, the method calculates the gradient of the error function with respect to the neural network's weights. It is a generalization of the delta rule for perceptrons to multilayer feedforward neural networks.

The "backwards" part of the name stems from the fact that calculation of the gradient proceeds backwards through the network, with the gradient of the final layer of weights being calculated first and the gradient of the first layer of weights being calculated last. Partial computations of the gradient from one layer are reused in the computation of the gradient for the previous layer. This backwards flow of the error information allows for efficient computation of the gradient at each layer versus the naive approach of calculating the gradient of each layer separately.

Backpropagation's popularity has experienced a recent resurgence given the widespread adoption of deep neural networks for image recognition and speech recognition. It is considered an efficient algorithm, and modern implementations take advantage of specialized GPUs to further improve performance.

__Approach__  

- Build a small neural network as defined in the architecture below.
- Initialize the weights and bias randomly.
- Fix the input and output.
- Forward pass the inputs. calculate the cost.
- compute the gradients and errors.
- Backprop and adjust the weights and bias accordingly

__Architecture:__

- Build a Feed Forward neural network with 2 hidden layers. All the layers will have 3 Neurons each.
- 1st and 2nd hidden layer will have Relu and sigmoid respectively as activation functions. Final layer will have Softmax.
- Error is calculated using cross-entropy.

<img src="./images/ann/backpropagation_network.png" width="100%"/>

### Initializing Network


<img src="http://latex.codecogs.com/svg.latex?Input = \begin{bmatrix} 0.1 & 0.2 & 0.7 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?W_{ij}  = \begin{bmatrix} W_{i1j1} & W_{i1j2} & W_{i1j3} \\ W_{i2j1} & W_{i2j2} & W_{i2j3} \\ W_{i3j1} & W_{i3j2} & W_{i3j3} \end{bmatrix} = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.3 & 0.2 & 0.7 \\ 0.4 & 0.3 & 0.9 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? W_{ij}  = \begin{bmatrix} W_{i1j1} & W_{i1j2} & W_{i1j3} \\ W_{i2j1} & W_{i2j2} & W_{i2j3} \\ W_{i3j1} & W_{i3j2} & W_{i3j3} \end{bmatrix} = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.3 & 0.2 & 0.7 \\ 0.4 & 0.3 & 0.9 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? W_{jk}  = \begin{bmatrix} W_{j1k1} & W_{j1k2} & W_{j1k3} \\ W_{j2k1} & W_{j2k2} & W_{j2k3} \\ W_{j3k1} & W_{j3k2} & W_{j3k3} \end{bmatrix}  = \begin{bmatrix} 0.2 & 0.3 & 0.5 \\ 0.3 & 0.5 & 0.7 \\ 0.6 & 0.4 & 0.8 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? W_{kl}  = \begin{bmatrix} W_{k1l1} & W_{k1l2} & W_{k1l3} \\ W_{k2l1} & W_{k2l2} & W_{k2l3} \\ W_{k3l1} & W_{k3l2} & W_{k3l3} \end{bmatrix}  = \begin{bmatrix} 0.1 & 0.4 & 0.8 \\ 0.3 & 0.7 & 0.2 \\ 0.5 & 0.2 & 0.9 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? Output = \begin{bmatrix} 1.0 & 0.0 & 0.0 \end{bmatrix}" border="0"/>

____________________________________________________________________________________________________________________________

<img src="./images/ann/backpropagation_network_l1.png" width="25%" style="float: right;"/>

__layer-1 Matrix Operation:__

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} i_1 & i_2 & i_3 \end{bmatrix} \times \begin{bmatrix} W_{i1j1} & W_{i1j2} & W_{i1j3} \\ W_{i2j1} & W_{i2j2} & W_{i2j3} \\ W_{i3j1} & W_{i3j2} & W_{i3j3} \end{bmatrix} + \begin{bmatrix} b_{j1} & b_{j2} & b_{j3} \end{bmatrix} = \begin{bmatrix} h1_{in1} & h1_{in2} & h1_{in3} \end{bmatrix}" border="0"/>

__layer-1 Relu Operation:__

<img src="http://latex.codecogs.com/svg.latex? relu = max(0,x) " border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} = \begin{bmatrix} max(0,h1_{in1}) & max(0,h1_{in2}) & max(0,h1_{in3}) \end{bmatrix} " border="0"/>

__layer-1 Example:__

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} 0.1 & 0.2 & 0.7 \end{bmatrix} \times \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.3 & 0.2 & 0.7 \\ 0.4 & 0.3 & 0.9 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix} " border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} = \begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix} " border="0"/>


________________________________________________________________________________________________________________________


<img src="./images/ann/backpropagation_network_l2.png" width="25%" style="float: right;"/>

__layer-2 Matrix Operation:__

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} \times \begin{bmatrix} W_{j1k1} & W_{j1k2} & W_{j1k3} \\ W_{j2k1} & W_{j2k2} & W_{j2k3} \\ W_{j3k1} & W_{j3k2} & W_{j3k3} \end{bmatrix} + \begin{bmatrix} b_{k1} & b_{k2} & b_{k3} \end{bmatrix} = \begin{bmatrix} h2_{in1} & h2_{in2} & h2_{in3} \end{bmatrix}" border="0"/>

__layer-2 Sigmoid Operation:__

<img src="http://latex.codecogs.com/svg.latex? Sigmoid = \frac{1}{1+e^{-x}} " border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} = \begin{bmatrix} 1/(1+e^{h2_{in1}}) & 1/(1+e^{h2_{in2}}) & 1/(1+e^{h2_{in3}}) \end{bmatrix}" border="0"/>

__layer-2 Example:__

<img src="http://latex.codecogs.com/svg.latex?\begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix}  \times \begin{bmatrix} 0.2 & 0.3 & 0.5 \\ 0.3 & 0.5 & 0.7 \\ 0.6 & 0.4 & 0.8 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 2.73 & 2.76 & 4.001 \end{bmatrix} " border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} = \begin{bmatrix} 0.938 & 0.94 & 0.98 \end{bmatrix} " border="0"/>

________________________________________________________________________________________________________________________


<img src="./images/ann/backpropagation_network_l3.png" width="25%" style="float: right;"/>

__layer-3 Matrix Operation:__

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} \times \begin{bmatrix} W_{k1l1} & W_{k1l2} & W_{k1l3} \\ W_{k2l1} & W_{k2l2} & W_{k2l3} \\ W_{k3l1} & W_{k3l2} & W_{k3l3} \end{bmatrix} + \begin{bmatrix} b_{l1} & b_{l2} & b_{l3} \end{bmatrix} = \begin{bmatrix} O_{in1} & O_{in2} & O_{in3} \end{bmatrix}" border="0"/>

__layer-3 Softmax Operation:__

<img src="http://latex.codecogs.com/svg.latex? Softmax =  e^{l_{ina}}/(\sum_{a=1}^3 e^{O_{ina}})" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} O_{out1} & O_{out2} & O_{out3} \end{bmatrix} = \begin{bmatrix} e^{O_{in1}}/(\sum_{a=1}^3 e^{O_{ina}}) & e^{O_{in2}}/(\sum_{a=1}^3 e^{O_{ina}}) & e^{O_{in3}}/(\sum_{a=1}^3 e^{O_{ina}}) \end{bmatrix}" border="0"/>

__layer-3 Example:__

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} 0.938 & 0.94 & 0.98 \end{bmatrix} \times \begin{bmatrix} 0.1 & 0.4 & 0.8 \\ 0.3 & 0.7 & 0.2 \\ 0.5 & 0.2 & 0.9 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.8658 & 2.2292 & 2.8204 \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} O_{out1} & O_{out2} & O_{out3} \end{bmatrix} = \begin{bmatrix} 0.19858 & 0.28559 & 0.51583 \end{bmatrix}" border="0"/>


________________________________________________________________________________________________________________________


## Analysis:

The Actual Output should be <img src="http://latex.codecogs.com/svg.latex?[1.0,\: 0.0,\: 0.0]" border="0"/> but we got <img src="http://latex.codecogs.com/svg.latex?[0.19858, \: 0.28559, \: 0.51583]" border="0"/>.  
To calculate error lets use cross-entropy

<img src="http://latex.codecogs.com/svg.latex? cross\: entropy = - (1/n)(\sum_{i=1}^{3} (y_{i} \times \log(O_{outi})) + ((1-y_{i}) \times \log((1-O_{outi}))))" border="0"/>

## Important Derivatives 

### Sigmoid

<img src="http://latex.codecogs.com/svg.latex?Sigmoid = 1/(1+\mathrm{e}^{-x})" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \frac{\partial (1/(1+\mathrm{e}^{-x}))}{\partial x} = 1/(1+\mathrm{e}^{-x}) \times (1- 1/(1+\mathrm{e}^{-x}))" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?\frac{\partial Sigmoid}{\partial x} = Sigmoid \times (1- Sigmoid)" border="0"/>
_____________________________________________________________________________________________________________________________

### Relu

<img src="http://latex.codecogs.com/svg.latex?relu = max(0, x) " border="0"/>

<img src="http://latex.codecogs.com/svg.latex?\ if x >0 , \frac{\partial (relu)}{\partial x} = 1" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?\ Otherwise , \frac{\partial (relu)}{\partial x} = 0" border="0"/>
_____________________________________________________________________________________________________________________________

### Softmax operation 


<img src="http://latex.codecogs.com/svg.latex?Softmax = \mathrm{e}^{x_{a}}/(\sum_{a=1}^{n}\mathrm{e}^{x_{a}}) = \mathrm{e}^{x_{1}}/(\mathrm{e}^{x_{1}} + \mathrm{e}^{x_{2}} + \mathrm{e}^{x_{3}}) " border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \frac{\partial (Softmax)}{\partial x_{1}} = (\mathrm{e}^{x_{1}} \times (\mathrm{e}^{x_{2}}+\mathrm{e}^{x_{3}}))/ (\mathrm{e}^{x_{1}}+\mathrm{e}^{x_{2}}+\mathrm{e}^{x_{3}})^2" border="0"/>

## BackPropagating the error - (Hidden Layer2 - Output Layer) Weights

<img src="./images/ann/backprop_layer3.png" width="95%" />

#### Lets calculate a few derviates upfront so these become handy and we can reuse them whenever necessary. 


<img src="http://latex.codecogs.com/svg.latex? \frac{\partial E_{total}}{\partial O_{out1}} = \frac{\partial (-1 * ((y_{1} * \log(O_{out1}) + (1-y_{1}) * \log((1-O_{out1}))}{\partial O_{out1}}" border="0"/>

#### Here are we are using only one example (batch_size=1), if there are more examples then just average everything.


<img src="http://latex.codecogs.com/svg.latex?\frac{\partial E_{1}}{\partial O_{out1}} =  -1 *  ((y_{1} * (1/O_{out1}) + (1-y_{1})*(1/(1-O_{out1}))" border="0"/>

#### by symmetry, We can calculate other derviatives also 

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial E_{1}}{\partial O_{out1}} \\ \frac{\partial E_{2}}{\partial O_{out2}} \\ \frac{\partial E_{3}}{\partial O_{out3}}  \end{bmatrix} = \begin{bmatrix} -1  * ((y_{1} * (1/O_{out1}) + (1-y_{1})*(1/(1-O_{out1})) \\ -1  * ((y_{2} * (1/O_{out2}) + (1-y_{2})*(1/(1-O_{out2})) \\ -1  * ((y_{3} * (1/O_{out3}) + (1-y_{3})*(1/(1-O_{out3})) \end{bmatrix}" border="0"/>

### In our Example The values will be

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial E_{1}}{\partial O_{out1}} \\ \frac{\partial E_{2}}{\partial O_{out2}}   \\ \frac{\partial E_{3}}{\partial O_{out3}} \end{bmatrix} =   \begin{bmatrix} -3.70644\\ -1.4755\\  -1.6886 \end{bmatrix} " border="0"/>



---

### Next let us calculate the derviative of each output with respect to their input.

<img src="http://latex.codecogs.com/svg.latex? \frac{\partial O_{out1}}{\partial O_{in1}} = \frac{\partial (\mathrm{e}^{O_{in1}}/(\mathrm{e}^{O_{in1}} + \mathrm{e}^{O_{in2}} + \mathrm{e}^{O_{in3}}))}{\partial O_{in1}}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \frac{\partial O_{out1}}{\partial O_{in1}} = (\mathrm{e}^{O_{in1}} \times (\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 " border="0"/>

#### by symmetry, We can calculate other derviatives also 
<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial O_{out1}}{\partial O_{in1}} \\ \frac{\partial O_{out2}}{\partial O_{in2}}  \\ \frac{\partial O_{out3}}{\partial O_{in3}}\\ \end{bmatrix}  =   \begin{bmatrix} (\mathrm{e}^{O_{in1}} \times (\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\ (\mathrm{e}^{O_{in2}} \times (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\ (\mathrm{e}^{O_{in3}} \times (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\ \end{bmatrix} " border="0"/>

### In our Example The values will be
<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial O_{out1}}{\partial O_{in1}} \\ \frac{\partial O_{out2}}{\partial O_{in2}}  \\ \frac{\partial O_{out3}}{\partial O_{in3}}\\ \end{bmatrix}  =   \begin{bmatrix} 0.15911 \\ 0.2040 \\ 0.3685 \\ \end{bmatrix}  " border="0"/>

#### For each input to neuron lets calculate the derivative with respect to each weight.

##### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex? \frac{\partial O_{in1}}{\partial W_{k1l1}} = \frac{\partial ((h2_{out1}*W_{j1k1}) + (h2_{out2} * W{j2k1}) + (h2_{out3} * W{j3k1}) + b_{l1})}{\partial W_{k1l1}}" border="0"/>

#### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex? \frac{\partial O_{in1}}{\partial W_{k1l1}} = h2_{out1} " border="0"/>

#### Using similarity we can write:
<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial O_{in1}}{\partial W_{k1l1}}  \\ \frac{\partial O_{in1}}{\partial W_{k2l1}}   \\ \frac{\partial O_{in1}}{\partial W_{k3l1}} \\ \end{bmatrix}  =   \begin{bmatrix} h2_{out1}\\ h2_{out2}\\ h2_{out3}\\ \end{bmatrix}  =  \begin{bmatrix} 0.938\\ 0.94\\ 0.98\\ \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial O_{in2}}{\partial W_{k1l2}}  \\ \frac{\partial O_{in2}}{\partial W_{k2l2}}   \\ \frac{\partial O_{in2}}{\partial W_{k3l2}} \\ \end{bmatrix}  =   \begin{bmatrix} h2_{out1}\\ h2_{out2}\\ h2_{out3}\\ \end{bmatrix}  =  \begin{bmatrix} 0.938\\ 0.94\\ 0.98\\ \end{bmatrix}" border="0"/>

<img src="http://latex.codecogs.com/svg.latex? \begin{bmatrix} \frac{\partial O_{in3}}{\partial W_{k1l3}}  \\ \frac{\partial O_{in3}}{\partial W_{k2l3}}   \\ \frac{\partial O_{in3}}{\partial W_{k3l3}} \\ \end{bmatrix}  =   \begin{bmatrix} h2_{out1}\\  h2_{out2}\\ h2_{out3}\\ \end{bmatrix}  =  \begin{bmatrix} 0.938\\ 0.94\\ 0.98\\ \end{bmatrix}" border="0"/>

#### Now we will calulate the change in 
<img src="http://latex.codecogs.com/svg.latex? W_{k1l1} " border="0"/>

#### This will be simply
<img src="http://latex.codecogs.com/svg.latex? \frac{\partial E_{1}}{\partial W_{k1l1}} " border="0"/>

#### Using chain rule:

<img src="http://latex.codecogs.com/svg.latex? \frac{\partial E_{1}}{\partial W_{k1l1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1l1}}" border="0"/>

### By symmetry
<img src="http://latex.codecogs.com/svg.latex? \delta W_{kl} =  \begin{bmatrix} \frac{\partial E_{1}}{\partial W_{k1l1}} & \frac{\partial E_{2}}{\partial W_{k1l2}} & \frac{\partial E_{3}}{\partial W_{k1l3}} \\ \frac{\partial E_{1}}{\partial W_{k2l1}} & \frac{\partial E_{2}}{\partial W_{k2l2}} & \frac{\partial E_{3}}{\partial W_{k2l3}} \\ \frac{\partial E_{1}}{\partial W_{k3l1}} & \frac{\partial E_{2}}{\partial W_{k3l2}} & \frac{\partial E_{3}}{\partial W_{k3l3}} \\ \end{bmatrix}  =   \begin{bmatrix} \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1l1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k1l2}}& \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k1l3}} \\ \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k2l1}}& \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k2l2}} & \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k2l3}} \\ \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k3l1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k3l2}} & \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k3l3}} \\ \end{bmatrix} " border="0"/>

#### All the above values are calculated above, We just need to substitute these values 
<img src="http://latex.codecogs.com/svg.latex?
\delta W_{kl} =  \begin{bmatrix}
\delta W_{k1l1} & \delta W_{k1l2} & \delta W_{k1l3} \\
\delta W_{k2l1} & \delta W_{k2l2} & \delta W_{k2l3} \\
\delta W_{k3l1} & \delta W_{k3l2} & \delta W_{k3l3} \\ \end{bmatrix}  =  \begin{bmatrix}
-3.7064 *0.1591 * 0.938&-0.301 * 0.204 * 0.938&-1.6886 * 0.3685 * 0.938 \\
-3.7064 * 0.1591 * 0.94&-0.301 * 0.204 * 0.94&-1.6886 * 0.3685 * 0.94 \\
-3.7064 * 0.1591* 0.98&-0.301  * 0.204 * 0.98&-1.6886  * 0.3685 * 0.98 \\ \end{bmatrix}  =   \begin{bmatrix}
-0.5531 & -0.0576 & -0.5836\\
-0.554347 & -0.0577 & -0.5849 \\
-0.577937 & -0.06017 & -0.6098 \\ \end{bmatrix} 
" border="0"/>

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{kl}} =  \begin{bmatrix}
W_{k1l1} - (lr*\delta W_{k1l1}) & W_{k1l2} - (lr * \delta W_{k1l2}) &W_{k1l3} - (lr * \delta W_{k1l3}) \\
W_{k2l1} - (lr* \delta W_{k2l1}) & W_{k2l2} - (lr * \delta W_{k2l2}) &W_{k2l3} - (lr * \delta W_{k2l3}) \\
W_{k3l1} - (lr * \delta W_{k3l1}) & W_{k3l2} - (lr *\delta W_{k3l2}) & W_{k3l3} - (lr * \delta W_{k3l3}) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{kl}} =  \begin{bmatrix}
0.1 - (0.01*-0.5531) & 0.4 - (0.01 * -0.0576) & 0.8 - (0.01 * -0.5836) \\
0.3 - (0.01* -0.554347) & 0.7 - (0.01* -0.0577) &0.2 - (0.01* -0.5849) \\
0.5 - (0.01 * -0.577937) & 0.2 - (0.01 -0.06017) & 0.9 - (0.01 * -0.6098) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{kl}} =  \begin{bmatrix}
0.105531 & 0.400576 & 0.805836 \\
0.30055 & 0.700577 &0.2005849 \\
0.5005779 & 0.2006017 & 0.9006098 \\ \end{bmatrix} 
" border="0"/>

#### Finally We made it, Lets jump to the next layer

## BackPropagating the error - (Hidden Layer1 - Hidden Layer2)  Weights

<img src="./images/ann/backprop_layer2.png" width="75%" />

#### Lets calculate a few handy derviatives before we actually calculate the error derviatives wrt Weights in this layer. 

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h2_{out1}}{\partial h2_{in1}} = \frac{\partial Sigmoid(h2_{in1})}{\partial h2_{in1}}
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h2_{out1}}{\partial h2_{in1}} = Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h2_{out1}}{\partial h2_{in1}} \\
\frac{\partial h2_{out2}}{\partial h2_{in2}}   \\
\frac{\partial h2_{out3}}{\partial h2_{in3}} \\ \end{bmatrix}  =   \begin{bmatrix}
 Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\
Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\
Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\ \end{bmatrix} 
" border="0"/>

#### In our example , this will be 
<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h2_{out1}}{\partial h2_{in1}} \\
\frac{\partial h2_{out2}}{\partial h2_{in2}}   \\
\frac{\partial h2_{out3}}{\partial h2_{in3}} \\ \end{bmatrix}  =   \begin{bmatrix}
 Sigmoid(2.73) * (1 - Sigmoid(2.73))\\
Sigmoid(2.76) * (1 - Sigmoid(2.76))\\
Sigmoid(4.001) * (1 - Sigmoid(4.001))\\ \end{bmatrix}  =   \begin{bmatrix}
 0.058156\\
0.0564\\
0.0196\\ \end{bmatrix}  
" border="0"/>

#### For each input to neuron lets calculate the derivative with respect to each weight.

##### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h2_{in1}}{\partial W_{j1k1}} = \frac{\partial ((h1_{out1}*W_{j1k1}) + (h1_{out2} * W{j2k1}) + (h1_{out3} * W{j3k1}) + b_{k1})}{\partial W_{j1k1}}
" border="0"/>

#### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h2_{in1}}{\partial W_{j1k1}} = h1_{out1}
" border="0"/>

#### Using similarity we can write:
<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h2_{in1}}{\partial W_{j1k1}}  \\
\frac{\partial h2_{in1}}{\partial W_{j2k1}}   \\
\frac{\partial h2_{in1}}{\partial W_{j3k1}} \\ \end{bmatrix}  =   \begin{bmatrix}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 1.35\\
 1.27\\
 1.8\\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h2_{in2}}{\partial W_{j1k2}}  \\
\frac{\partial h2_{in2}}{\partial W_{j2k2}}   \\
\frac{\partial h2_{in2}}{\partial W_{j3k2}} \\ \end{bmatrix}  =   \begin{bmatrix}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 1.35\\
 1.27\\
 1.8\\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h2_{in3}}{\partial W_{j1k3}}  \\
\frac{\partial h2_{in3}}{\partial W_{j2k3}}   \\
\frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{bmatrix}  =   \begin{bmatrix}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 1.35\\
 1.27\\
 1.8\\ \end{bmatrix} 
" border="0"/>

##### Now we will calulate the change in 
<img src="http://latex.codecogs.com/svg.latex?
W_{j3k1}
" border="0"/> 
##### and generalize it for all other variables.

#### Caution: Make sure that you have understood everything we discussed till here. 

##### This will be simply
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{total}}{\partial W_{j3k1}}
" border="0"/>

##### Using chain rule:
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{total}}{\partial W_{j3k1}} = \frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j3k1}}
" border="0"/>

##### Now we will see each and every equation individually.


#### Lets look at the matrix
### By symmetry
<img src="http://latex.codecogs.com/svg.latex?
\delta W_{jk} =  \begin{bmatrix}
\frac{\partial E_{total}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial W_{j1k2}} & \frac{\partial E_{total}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial W_{j2k1}} & \frac{\partial E_{total}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial W_{j3k3}} \\ \end{bmatrix}  =   \begin{bmatrix}
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j1k2}}& \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j2k1}}& \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{bmatrix} 
" border="0"/>

#### We have Already calculated the 2nd and 3rd term in each matrix. We need to check on the 1st term. If we see the matrix, the first term is common in all the columns. So there are only three values. Lets look into one value

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{total}}{\partial h2_{out1}} = \frac{\partial E_{1}}{\partial h2_{out1}} + \frac{\partial E_{2}}{\partial h2_{out1}} + \frac{\partial E_{3}}{\partial h2_{out1}}
" border="0"/>

#### Lets see what each individual term boils down too.

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{1}}{\partial h2_{out1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out1}}
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{2}}{\partial h2_{out1}} = \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out1}}
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{3}}{\partial h2_{out1}} = \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out1}}
" border="0"/>

### BY symmentry

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out1}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out1}}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out1}})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out2}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out2}}) + (\frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out2}})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out3}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out3}}) + (\frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out3}})\\ \end{bmatrix} 
" border="0"/>

__Again the first two values are already calculated by us when dealing with derviatives of W_{kl}. We just need to calculate the third one, Which is the derivative of  input to each output layer wrt output of hidden layer-2. It is nothing but the corresponding weight which connects both the layers.__

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial O_{inl}}{\partial h2_{out1}} & \frac{\partial O_{in2}}{\partial h2_{out1}} & \frac{\partial O_{in3}}{\partial h2_{out1}} \\
\frac{\partial O_{inl}}{\partial h2_{out2}} & \frac{\partial O_{in2}}{\partial h2_{out2}} & \frac{\partial O_{in3}}{\partial h2_{out2}}\\
\frac{\partial O_{inl}}{\partial h2_{out3}} & \frac{\partial O_{in2}}{\partial h2_{out3}} & \frac{\partial O_{in3}}{\partial h2_{out3}} \\ \end{bmatrix}  =  \begin{bmatrix}
W_{k1l1} & W_{k1l2} & W_{k1l3} \\
W_{k2l1} & W_{k2l2} & W_{k2l3} \\
W_{k3l1} & W_{k3l2} & W_{k3l3} \\ \end{bmatrix}  
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k1l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k1l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k1l3})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k2l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k2l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k2l3})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k3l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k3l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k2l3})\\
\end{bmatrix} 
" border="0"/>

#### All Values are calculated before we just need to impute the corresponding values for our example.
<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(-3.70644 *0.15911 * 0.1) + (-1.4755 * 0.2040 *0.4) + ( -1.6886 * 0.3685 * 0.8)\\
(-3.70644 * 0.15911 * 0.3) + (-1.4755 * 0.2040 * 0.7) + ( -1.6886 * 0.3685 * 0.2)\\
(-3.70644 * 0.15911  * 0.5) + (-1.4755 * 0.2040 * 0.2) + ( -1.6886 * 0.3685 * 0.9)\\
\end{bmatrix}  =   \begin{bmatrix}
(-0.0589) + (-0.2383) + ( -0.5931)\\
(-0.1769) + (-0.417) + ( -0.14828)\\
(-0.2948) + (-0.119) + ( -0.667)\\
\end{bmatrix}  =
 \begin{bmatrix}
-0.8903\\
-0.74218\\
-1.0810\\
\end{bmatrix} 
" border="0"/>

#### Lets look at the matrix
### By symmetry
<img src="http://latex.codecogs.com/svg.latex?
\delta W_{jk} =  \begin{bmatrix}
\frac{\partial E_{total}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial W_{j1k2}} & \frac{\partial E_{total}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial W_{j2k1}} & \frac{\partial E_{total}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial W_{j3k3}} \\ \end{bmatrix}  =   \begin{bmatrix}
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j1k2}}& \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j2k1}}& \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2
_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\delta W_{jk} =   \begin{bmatrix}
-0.8903 * 0.058156 * 1.35 & -0.74218 * 0.0564 * 1.35 & -1.0810 * 0.0196 * 1.35 \\
-0.8903 * 0.058156  * 1.27 & -0.74218 * 0.0564 * 1.27 & -1.0810 * 0.0196 * 1.27 \\
-0.8903 * 0.058156 * 1.8 & -0.74218 * 0.0564 * 1.8 & -1.0810 * 0.0196 * 1.8 \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\delta W_{jk} =   \begin{bmatrix}
-0.06989 & -0.0565 & -0.0286 \\
-0.06575 & -0.05316 & -0.0269 \\
-0.0932 & -0.0753 & -0.03813 \\ \end{bmatrix} 
" border="0"/>

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{jk}} =  \begin{bmatrix}
W_{j1k1} - (lr*\delta W_{j1k1}) & W_{j1k2} - (lr * \delta W_{j1k2}) &W_{j1k3} - (lr * \delta W_{j1k3}) \\
W_{j2k1} - (lr* \delta W_{j2k1}) & W_{j2k2} - (lr * \delta W_{j2k2}) &W_{j2k3} - (lr * \delta W_{j2k3}) \\
W_{j3k1} - (lr * \delta W_{j3k1}) & W_{j3k2} - (lr *\delta W_{j3k2}) & W_{j3k3} - (lr * \delta W_{j3k3}) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{jk}} =  \begin{bmatrix}
0.2 - (0.01*-0.06989) & 0.3 - (0.01 * -0.0565) & 0.5 - (0.01 * -0.0286) \\
0.3 - (0.01* -0.06575) & 0.5 - (0.01* -0.05316) &0.7 - (0.01* -0.0269) \\
0.6 - (0.01 * -0.0932) & 0.4 - (0.01 -0.0753) & 0.8 - (0.01 * -0.03813) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{jk}} =  \begin{bmatrix}
0.2006989 & 0.300565 & 0.500286 \\
0.3006575 & 0.5005316 &0.700269 \\
0.600932 & 0.400753 & 0.803813 \\ \end{bmatrix} 
" border="0"/>

#### Finally We made it, Lets jump to the next layer

## BackPropagating the error - (Input Layer - Hidden Layer1)  Weights

<img src="./images/ann/backprop_layer1.png" width="75%" />

#### Lets calculate a few handy derviatives before we actually calculate the error derviatives wrt Weights in this layer. 

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h1_{out1}}{\partial h1_{in1}} = \frac{\partial Relu(h1_{in1})}{\partial h1_{in1}}
" border="0"/>

#### We already know 
### Relu
<img src="http://latex.codecogs.com/svg.latex?
relu = max(0, x) 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\ if x >0 , \frac{\partial (relu)}{\partial x} = 1
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\ Otherwise , \frac{\partial (relu)}{\partial x} = 0
" border="0"/>

### Since the inputs are positive

<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h1_{out1}}{\partial h1_{in1}} = 1.0
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h1_{out1}}{\partial h1_{in1}} \\
\frac{\partial h1_{out2}}{\partial h1_{in2}}   \\
\frac{\partial h1_{out3}}{\partial h1_{in3}} \\ \end{bmatrix}  =   \begin{bmatrix}
 1.0\\
1.0\\
1.0\\ \end{bmatrix} 
" border="0"/>


#### For each input to neuron lets calculate the derivative with respect to each weight.

#### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h1_{in1}}{\partial W_{i1j1}} = \frac{\partial ((I_{out1}*W_{j1k1}) + (I_{out2} * W{i2j1}) + (I_{out3} * W{i3j1}) + b_{j1})}{\partial W_{i1j1}}
" border="0"/>

#### Now let us look at the final derivative
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial h1_{in1}}{\partial W_{i1j1}} = I_{out1}
" border="0"/>

#### Using similarity we can write:
<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h1_{in1}}{\partial W_{i1j1}}  \\
\frac{\partial h1_{in1}}{\partial W_{i2j1}}   \\
\frac{\partial h1_{in1}}{\partial W_{i3j1}} \\ \end{bmatrix}  =   \begin{bmatrix}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 0.1\\
 0.2\\
 0.7\\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h1_{in2}}{\partial W_{i1j2}}  \\
\frac{\partial h1_{in2}}{\partial W_{i2j2}}   \\
\frac{\partial h1_{in2}}{\partial W_{i3j2}} \\ \end{bmatrix}  =   \begin{bmatrix}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 0.1\\
 0.2\\
 0.7\\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial h1_{in3}}{\partial W_{i1j3}}  \\
\frac{\partial h1_{in3}}{\partial W_{i2j3}}   \\
\frac{\partial h1_{in3}}{\partial W_{i3j3}} \\ \end{bmatrix}  =   \begin{bmatrix}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{bmatrix}  =  \begin{bmatrix}
 0.1\\
 0.2\\
 0.7\\ \end{bmatrix} 
" border="0"/>

##### Now we will calulate the change in 
<img src="http://latex.codecogs.com/svg.latex?
W_{j3k1}
" border="0"/>

##### and generalize it for all other variables.

#### Caution: Make sure that you have understood everything we discussed till here. 

##### This will be simply
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{total}}{\partial W_{i2j1}}
" border="0"/>

##### Using chain rule:
<img src="http://latex.codecogs.com/svg.latex?
\frac{\partial E_{total}}{\partial W_{i2j1}} = \frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1
_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i2j1}}
" border="0"/>

##### Now we will see each and every equation individually.

#### Lets look at the matrix
### By symmetry
<img src="http://latex.codecogs.com/svg.latex?
\delta W_{ij} =  \begin{bmatrix}
\frac{\partial E_{total}}{\partial W_{i1j1}} & \frac{\partial E_{total}}{\partial W_{i1j2}} & \frac{\partial E_{total}}{\partial W_{i1j3}} \\
\frac{\partial E_{total}}{\partial W_{i2j1}} & \frac{\partial E_{total}}{\partial W_{i2j2}} & \frac{\partial E_{total}}{\partial W_{i2j3}} \\
\frac{\partial E_{total}}{\partial W_{i3j1}} & \frac{\partial E_{total}}{\partial W_{i3j2}} & \frac{\partial E_{total}}{\partial W_{i3j3}} \\ \end{bmatrix}  =   \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i1j1}} & \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h1
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i1j2}}& \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h1
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i1j3}} \\
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i2j1}}& \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h2
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i2j2}} & \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h2
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i2j3}} \\
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i3j1}} & \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h2
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i3j2}} & \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h2
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i3j3}} \\ \end{bmatrix} 
" border="0"/>

#### We know the 2nd and 3rd derivatives in each cell in the above matrix. Lets look at how to get to derivative of 1st term in each cell.

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial h1_{out1}}) \\
(\frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial h1_{out2}})\\
(\frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial h1_{out3}})\\ \end{bmatrix} 
" border="0"/>

### We have calculated all the values previously except the last one in each cell, which is a simple derivative of linear terms.

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * W_{j1k1} \\
(\frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2_{out2}}{\partial h2_{in2}} * W_{j2k2}\\
(\frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2_{out3}}{\partial h2_{in3}} * W_{j3k3}\\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
-0.8903 *0.058156 * 0.2 \\
-0.74218 * 0.0564 *0.5\\
-1.0810 * 0.0196 * 0.8\\ \end{bmatrix}  =   \begin{bmatrix}
-0.01035 \\
-0.0209\\
-0.0169\\ \end{bmatrix} " border="0"/>

#### Lets look at the matrix
### By symmetry
<img src="http://latex.codecogs.com/svg.latex?
\delta W_{ij} =  \begin{bmatrix}
\frac{\partial E_{total}}{\partial W_{i1j1}} & \frac{\partial E_{total}}{\partial W_{i1j2}} & \frac{\partial E_{total}}{\partial W_{i1j3}} \\
\frac{\partial E_{total}}{\partial W_{i2j1}} & \frac{\partial E_{total}}{\partial W_{i2j2}} & \frac{\partial E_{total}}{\partial W_{i2j3}} \\
\frac{\partial E_{total}}{\partial W_{i3j1}} & \frac{\partial E_{total}}{\partial W_{i3j2}} & \frac{\partial E_{total}}{\partial W_{i3j3}} \\ \end{bmatrix}  =   \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i1j1}} & \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h1
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i1j2}}& \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h1
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i1j3}} \\
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i2j1}}& \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h2
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i2j2}} & \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h2
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i2j3}} \\
\frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i3j1}} & \frac{\partial E_{total}}{\partial h1_{out2}} * \frac{\partial h2
_{out2}}{\partial h1_{in2}} * \frac{\partial h1_{in2}}{\partial W_{i3j2}} & \frac{\partial E_{total}}{\partial h1_{out3}} * \frac{\partial h2
_{out3}}{\partial h1_{in3}} * \frac{\partial h1_{in3}}{\partial W_{i3j3}} \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\delta W_{ij} =   \begin{bmatrix}
-0.01035 * 1 * 0.1 & -0.0209 * 1 * 0.1 & -0.0169 * 1 * 0.1 \\
-0.01035 * 1  * 0.2 & -0.0209* 1 * 0.2 & -0.0169 * 1 * 0.2 \\
-0.01035 * 1 * 0.7& -0.0209 * 1 * 0.7 & -0.0169 * 1 * 0.7 \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\delta W_{ij} =   \begin{bmatrix}
-0.001035 & -0.00209 & -0.00169 \\
-0.00207 & -0.00418 & -0.00338 \\
-0.007245 & -0.01463 & -0.01183 \\ \end{bmatrix} 
" border="0"/>

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{ij}} =  \begin{bmatrix}
W_{i1j1} - (lr*\delta W_{i1j1}) & W_{i1j2} - (lr * \delta W_{i1j2}) &W_{i1j3} - (lr * \delta W_{i1j3}) \\
W_{i2j1} - (lr* \delta W_{i2j1}) & W_{i2j2} - (lr * \delta W_{i2j2}) &W_{i2j3} - (lr * \delta W_{i2j3}) \\
W_{i3j1} - (lr * \delta W_{i3j1}) & W_{i3j2} - (lr *\delta W_{i3j2}) & W_{i3j3} - (lr * \delta W_{i3j3}) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{ij}} =  \begin{bmatrix}
0.1 - (0.01*-0.001035) & 0.2 - (0.01 * -0.00209) & 0.3 - (0.01 * -0.00169) \\
0.3 - (0.01* -0.00207) & 0.2 - (0.01* -0.00418) &0.7 - (0.01* -0.00338) \\
0.4 - (0.01 * -0.0007245) & 0.3 - (0.01 -0.01463) & 0.9 - (0.01 * -0.01183) \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{ij}} =  \begin{bmatrix}
0.10001035 & 0.2000209 & 0.3000169 \\
0.3000207 & 0.2000418 &0.7000338 \\
0.40007245 & 0.3001463 & 0.9001183 \\ \end{bmatrix} 
" border="0"/>

#### The End 

## Our Inital Weights 


<img src="http://latex.codecogs.com/svg.latex?
W_{ij} =  \begin{bmatrix}
W_{i1j1} & W_{i1j2} & W_{i1j3} \\
W_{i2j1} & W_{i2j2} & W_{i2j3} \\
W_{i3j1} & W_{i3j2} & W_{i3j3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.3 & 0.2 & 0.7 \\
0.4 & 0.3 & 0.9 \\ \end{bmatrix} 
" border="0"/>


<img src="http://latex.codecogs.com/svg.latex?
W_{jk} =  \begin{bmatrix}
W_{j1k1} & W_{j1k2} & W_{j1k3} \\
W_{j2k1} & W_{j2k2} & W_{j2k3} \\
W_{j3k1} & W_{j3k2} & W_{j3k3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.2 & 0.3 & 0.5 \\
0.3 & 0.5 & 0.7 \\
0.6 & 0.4 & 0.8 \\ \end{bmatrix} 
" border="0"/>


<img src="http://latex.codecogs.com/svg.latex?
W_{kl} =  \begin{bmatrix}
W_{k1l1} & W_{k1l2} & W_{k1l3} \\
W_{k2l1} & W_{k2l2} & W_{k2l3} \\
W_{k3l1} & W_{k3l2} & W_{k3l3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.1 & 0.4 & 0.8 \\
0.3 & 0.7 & 0.2 \\
0.5 & 0.2 & 0.9 \\ \end{bmatrix} 
" border="0"/>

## Our final weights 
<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{ij}} =  \begin{bmatrix}
0.10001035 & 0.2000209 & 0.3000169 \\
0.3000207 & 0.2000418 &0.7000338 \\
0.40007245 & 0.3001463 & 0.9001183 \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{jk}} =  \begin{bmatrix}
0.2006989 & 0.300565 & 0.500286 \\
0.3006575 & 0.5005316 &0.700269 \\
0.600932 & 0.400753 & 0.803813 \\ \end{bmatrix} 
" border="0"/>

<img src="http://latex.codecogs.com/svg.latex?
\acute{W_{kl}} =  \begin{bmatrix}
0.105531 & 0.400576 & 0.805836 \\
0.30055 & 0.700577 &0.2005849 \\
0.5005779 & 0.2006017 & 0.9006098 \\ \end{bmatrix} 
" border="0"/>

## Important Notes:
- I have completely eliminated bias when differentiating. Do you know why ?
- Backprop of bias should be straight forward. Try on your own.
- I have taken only one example. What will happen if we take batch of examples?
- Though I have not mentioned directly about vansihing gradients. Do you see why it occurs?
- What would happen if all the weights are the same number instead of random ?
