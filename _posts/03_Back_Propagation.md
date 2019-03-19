
# Back propagation

https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

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

<img src="./images/backpropagation_network.png" width="100%"/>

### Initializing Network

$$ Input = \begin{bmatrix} 0.1 & 0.2 & 0.7 \end{bmatrix}$$

$$ W_{ij}  = \begin{bmatrix} W_{i1j1} & W_{i1j2} & W_{i1j3} \\ W_{i2j1} & W_{i2j2} & W_{i2j3} \\ W_{i3j1} & W_{i3j2} & W_{i3j3} \end{bmatrix} = \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.3 & 0.2 & 0.7 \\ 0.4 & 0.3 & 0.9 \end{bmatrix}$$

$$ W_{jk}  = \begin{bmatrix} W_{j1k1} & W_{j1k2} & W_{j1k3} \\ W_{j2k1} & W_{j2k2} & W_{j2k3} \\ W_{j3k1} & W_{j3k2} & W_{j3k3} \end{bmatrix}  = \begin{bmatrix} 0.2 & 0.3 & 0.5 \\ 0.3 & 0.5 & 0.7 \\ 0.6 & 0.4 & 0.8 \end{bmatrix} $$

$$ W_{kl}  = \begin{bmatrix} W_{k1l1} & W_{k1l2} & W_{k1l3} \\ W_{k2l1} & W_{k2l2} & W_{k2l3} \\ W_{k3l1} & W_{k3l2} & W_{k3l3} \end{bmatrix}  = \begin{bmatrix} 0.1 & 0.4 & 0.8 \\ 0.3 & 0.7 & 0.2 \\ 0.5 & 0.2 & 0.9 \end{bmatrix}$$

$$ Output = \begin{bmatrix} 1.0 & 0.0 & 0.0 \end{bmatrix}$$

____________________________________________________________________________________________________________________________

<img src="./images/backpropagation_network_l1.png" width="25%" style="float: right;"/>

__layer-1 Matrix Operation:__

$ \begin{bmatrix} i_1 & i_2 & i_3 \end{bmatrix} \times \begin{bmatrix} W_{i1j1} & W_{i1j2} & W_{i1j3} \\ W_{i2j1} & W_{i2j2} & W_{i2j3} \\ W_{i3j1} & W_{i3j2} & W_{i3j3} \end{bmatrix} + \begin{bmatrix} b_{j1} & b_{j2} & b_{j3} \end{bmatrix} = \begin{bmatrix} h1_{in1} & h1_{in2} & h1_{in3} \end{bmatrix}$

__layer-1 Relu Operation:__

$ relu = max(0,x) $

$ \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} = \begin{bmatrix} max(0,h1_{in1}) & max(0,h1_{in2}) & max(0,h1_{in3}) \end{bmatrix} $

__layer-1 Example:__

$\begin{bmatrix} 0.1 & 0.2 & 0.7 \end{bmatrix} \times \begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.3 & 0.2 & 0.7 \\ 0.4 & 0.3 & 0.9 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix} $

$ \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} = \begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix} $


________________________________________________________________________________________________________________________


<img src="./images/backpropagation_network_l2.png" width="25%" style="float: right;"/>

__layer-2 Matrix Operation:__

$ \begin{bmatrix} h1_{out1} & h1_{out2} & h1_{out3} \end{bmatrix} \times \begin{bmatrix} W_{j1k1} & W_{j1k2} & W_{j1k3} \\ W_{j2k1} & W_{j2k2} & W_{j2k3} \\ W_{j3k1} & W_{j3k2} & W_{j3k3} \end{bmatrix} + \begin{bmatrix} b_{k1} & b_{k2} & b_{k3} \end{bmatrix} = \begin{bmatrix} h2_{in1} & h2_{in2} & h2_{in3} \end{bmatrix}$

__layer-2 Sigmoid Operation:__

$ Sigmoid = \frac{1}{1+e^{-x}} $

$ \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} = \begin{bmatrix} 1/(1+e^{h2_{in1}}) & 1/(1+e^{h2_{in2}}) & 1/(1+e^{h2_{in3}}) \end{bmatrix} $

__layer-2 Example:__

$\begin{bmatrix} 1.35 & 1.27 & 1.8 \end{bmatrix}  \times \begin{bmatrix} 0.2 & 0.3 & 0.5 \\ 0.3 & 0.5 & 0.7 \\ 0.6 & 0.4 & 0.8 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 2.73 & 2.76 & 4.001 \end{bmatrix} $

$ \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} = \begin{bmatrix} 0.938 & 0.94 & 0.98 \end{bmatrix} $

________________________________________________________________________________________________________________________


<img src="./images/backpropagation_network_l3.png" width="25%" style="float: right;"/>



__layer-3 Matrix Operation:__



$ \begin{bmatrix} h2_{out1} & h2_{out2} & h2_{out3} \end{bmatrix} \times \begin{bmatrix} W_{k1l1} & W_{k1l2} & W_{k1l3} \\ W_{k2l1} & W_{k2l2} & W_{k2l3} \\ W_{k3l1} & W_{k3l2} & W_{k3l3} \end{bmatrix} + \begin{bmatrix} b_{l1} & b_{l2} & b_{l3} \end{bmatrix} = \begin{bmatrix} O_{in1} & O_{in2} & O_{in3} \end{bmatrix}$



__layer-3 Softmax Operation:__



$ Softmax =  e^{l_{ina}}/(\sum_{a=1}^3 e^{O_{ina}}) $

$ \begin{bmatrix} O_{out1} & O_{out2} & O_{out3} \end{bmatrix} = \begin{bmatrix} e^{O_{in1}}/(\sum_{a=1}^3 e^{O_{ina}}) & e^{O_{in2}}/(\sum_{a=1}^3 e^{O_{ina}}) & e^{O_{in3}}/(\sum_{a=1}^3 e^{O_{ina}}) \end{bmatrix} $



__layer-3 Example:__



$ \begin{bmatrix} 0.938 & 0.94 & 0.98 \end{bmatrix} \times \begin{bmatrix} 0.1 & 0.4 & 0.8 \\ 0.3 & 0.7 & 0.2 \\ 0.5 & 0.2 & 0.9 \end{bmatrix} + \begin{bmatrix} 1.0 & 1.0 & 1.0 \end{bmatrix} = \begin{bmatrix} 1.8658 & 2.2292 & 2.8204 \end{bmatrix} $


$ \begin{bmatrix} O_{out1} & O_{out2} & O_{out3} \end{bmatrix} = \begin{bmatrix} 0.19858 & 0.28559 & 0.51583 \end{bmatrix} $


________________________________________________________________________________________________________________________


## Analysis:

The Actual Output should be $[1.0,\: 0.0,\: 0.0]$ but we got $[0.19858, \: 0.28559, \: 0.51583]$.  
To calculate error lets use cross-entropy




$$ cross\: entropy = - (1/n)(\sum_{i=1}^{3} (y_{i} \times \log(O_{outi})) + ((1-y_{i}) \times \log((1-O_{outi}))))$$

## Important Derivatives 

### Sigmoid

$$Sigmoid = 1/(1+\mathrm{e}^{-x})$$


$$ \frac{\partial (1/(1+\mathrm{e}^{-x}))}{\partial x} = 1/(1+\mathrm{e}^{-x}) \times (1- 1/(1+\mathrm{e}^{-x}))$$

$$\frac{\partial Sigmoid}{\partial x} = Sigmoid \times (1- Sigmoid)$$
_____________________________________________________________________________________________________________________________

### Relu

$$relu = max(0, x) $$

$$\ if x >0 , \frac{\partial (relu)}{\partial x} = 1$$

$$\ Otherwise , \frac{\partial (relu)}{\partial x} = 0$$
_____________________________________________________________________________________________________________________________

### Softmax operation 


$$Softmax = \mathrm{e}^{x_{a}}/(\sum_{a=1}^{n}\mathrm{e}^{x_{a}}) = \mathrm{e}^{x_{1}}/(\mathrm{e}^{x_{1}} + \mathrm{e}^{x_{2}} + \mathrm{e}^{x_{3}}) $$

$$ \frac{\partial (Softmax)}{\partial x_{1}} = (\mathrm{e}^{x_{1}} \times (\mathrm{e}^{x_{2}}+\mathrm{e}^{x_{3}}))/ (\mathrm{e}^{x_{1}}+\mathrm{e}^{x_{2}}+\mathrm{e}^{x_{3}})^2$$

## BackPropagating the error - (Hidden Layer2 - Output Layer) Weights

<img src="./images/backprop_layer3.png" width="95%" />

#### Lets calculate a few derviates upfront so these become handy and we can reuse them whenever necessary. 


$$ \frac{\partial E_{total}}{\partial O_{out1}} = \frac{\partial (-1 * ((y_{1} * \log(O_{out1}) + (1-y_{1}) * \log((1-O_{out1}))}{\partial O_{out1}} $$

#### Here are we are using only one example (batch_size=1), if there are more examples then just average everything.


$$\frac{\partial E_{1}}{\partial O_{out1}} =  -1 *  ((y_{1} * (1/O_{out1}) + (1-y_{1})*(1/(1-O_{out1}))$$

#### by symmetry, We can calculate other derviatives also 

$$ \begin{bmatrix} \frac{\partial E_{1}}{\partial O_{out1}} 
\\ \frac{\partial E_{2}}{\partial O_{out2}}  
\\ \frac{\partial E_{3}}{\partial O_{out3}}  \end{bmatrix} = \begin{bmatrix} -1  * ((y_{1} * (1/O_{out1}) + (1-y_{1})*(1/(1-O_{out1})) \\ -1  * ((y_{2} * (1/O_{out2}) + (1-y_{2})*(1/(1-O_{out2})) \\ -1  * ((y_{3} * (1/O_{out3}) + (1-y_{3})*(1/(1-O_{out3})) \end{bmatrix}
$$


### In our Example The values will be


$$ \begin{bmatrix} \frac{\partial E_{1}}{\partial O_{out1}} \\ \frac{\partial E_{2}}{\partial O_{out2}}   \\ \frac{\partial E_{3}}{\partial O_{out3}} \end{bmatrix} =   \begin{bmatrix} -3.70644\\ -1.4755\\  -1.6886 \end{bmatrix} $$



---

### Next let us calculate the derviative of each output with respect to their input.

\begin{equation*}
\frac{\partial O_{out1}}{\partial O_{in1}} = \frac{\partial (\mathrm{e}^{O_{in1}}/(\mathrm{e}^{O_{in1}} + \mathrm{e}^{O_{in2}} + \mathrm{e}^{O_{in3}}))}{\partial O_{in1}}  
\end{equation*}


\begin{equation*}
\frac{\partial O_{out1}}{\partial O_{in1}} = (\mathrm{e}^{O_{in1}} \times (\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2
\end{equation*}

#### by symmetry, We can calculate other derviatives also 

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{out1}}{\partial O_{in1}} \\
\frac{\partial O_{out2}}{\partial O_{in2}}  \\
\frac{\partial O_{out3}}{\partial O_{in3}}\\ \end{array} \right] =  \left[ \begin{array}{cccc}
(\mathrm{e}^{O_{in1}} \times (\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\
(\mathrm{e}^{O_{in2}} \times (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in3}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\
(\mathrm{e}^{O_{in3}} \times (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}))/ (\mathrm{e}^{O_{in1}}+\mathrm{e}^{O_{in2}}+\mathrm{e}^{O_{in3}})^2 \\ \end{array} \right]
\end{equation*}

### In our Example The values will be
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{out1}}{\partial O_{in1}} \\
\frac{\partial O_{out2}}{\partial O_{in2}}  \\
\frac{\partial O_{out3}}{\partial O_{in3}}\\ \end{array} \right] =  \left[ \begin{array}{cccc}
 0.15911\\
 0.2040\\
 0.3685\\ \end{array} \right]
\end{equation*}


#### For each input to neuron lets calculate the derivative with respect to each weight.

##### Now let us look at the final derivative
\begin{equation*}
\frac{\partial O_{in1}}{\partial W_{k1l1}} = \frac{\partial ((h2_{out1}*W_{j1k1}) + (h2_{out2} * W{j2k1}) + (h2_{out3} * W{j3k1}) + b_{l1})}{\partial W_{k1l1}}
\end{equation*}

#### Now let us look at the final derivative
\begin{equation*}
\frac{\partial O_{in1}}{\partial W_{k1l1}} = h2_{out1}
\end{equation*}

#### Using similarity we can write:
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{in1}}{\partial W_{k1l1}}  \\
\frac{\partial O_{in1}}{\partial W_{k2l1}}   \\
\frac{\partial O_{in1}}{\partial W_{k3l1}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h2_{out1}\\
 h2_{out2}\\
 h2_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.938\\
 0.94\\
 0.98\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{in2}}{\partial W_{k1l2}}  \\
\frac{\partial O_{in2}}{\partial W_{k2l2}}   \\
\frac{\partial O_{in2}}{\partial W_{k3l2}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h2_{out1}\\
 h2_{out2}\\
 h2_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.938\\
 0.94\\
 0.98\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{in3}}{\partial W_{k1l3}}  \\
\frac{\partial O_{in3}}{\partial W_{k2l3}}   \\
\frac{\partial O_{in3}}{\partial W_{k3l3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h2_{out1}\\
 h2_{out2}\\
 h2_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.938\\
 0.94\\
 0.98\\ \end{array} \right]
\end{equation*}

#### Now we will calulate the change in 
\begin{equation*}
W_{k1l1}
\end{equation*}

#### This will be simply
\begin{equation*}
\frac{\partial E_{1}}{\partial W_{k1l1}}
\end{equation*}

#### Using chain rule:
\begin{equation*}
\frac{\partial E_{1}}{\partial W_{k1l1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1l1}}
\end{equation*}

### By symmetry
\begin{equation*}
\delta W_{kl} = \left[ \begin{array}{cccc}
\frac{\partial E_{1}}{\partial W_{k1l1}} & \frac{\partial E_{2}}{\partial W_{k1l2}} & \frac{\partial E_{3}}{\partial W_{k1l3}} \\
\frac{\partial E_{1}}{\partial W_{k2l1}} & \frac{\partial E_{2}}{\partial W_{k2l2}} & \frac{\partial E_{3}}{\partial W_{k2l3}} \\
\frac{\partial E_{1}}{\partial W_{k3l1}} & \frac{\partial E_{2}}{\partial W_{k3l2}} & \frac{\partial E_{3}}{\partial W_{k3l3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k1l1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k1l2}}& \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O
_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k1l3}} \\
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k2l1}}& \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k2l2}} & \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O
_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k2l3}} \\
\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O
_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial W_{k3l1}} & \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O
_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial W_{k3l2}} & \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O
_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial W_{k3l3}} \\ \end{array} \right]
\end{equation*}

#### All the above values are calculated above, We just need to substitute these values 
\begin{equation*}
\delta W_{kl} = \left[ \begin{array}{cccc}
\delta W_{k1l1} & \delta W_{k1l2} & \delta W_{k1l3} \\
\delta W_{k2l1} & \delta W_{k2l2} & \delta W_{k2l3} \\
\delta W_{k3l1} & \delta W_{k3l2} & \delta W_{k3l3} \\ \end{array} \right] = \left[ \begin{array}{cccc}
-3.7064 *0.1591 * 0.938&-0.301 * 0.204 * 0.938&-1.6886 * 0.3685 * 0.938 \\
-3.7064 * 0.1591 * 0.94&-0.301 * 0.204 * 0.94&-1.6886 * 0.3685 * 0.94 \\
-3.7064 * 0.1591* 0.98&-0.301  * 0.204 * 0.98&-1.6886  * 0.3685 * 0.98 \\ \end{array} \right] =  \left[ \begin{array}{cccc}
-0.5531 & -0.0576 & -0.5836\\
-0.554347 & -0.0577 & -0.5849 \\
-0.577937 & -0.06017 & -0.6098 \\ \end{array} \right]
\end{equation*}

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
\begin{equation*}
\acute{W_{kl}} = \left[ \begin{array}{cccc}
W_{k1l1} - (lr*\delta W_{k1l1}) & W_{k1l2} - (lr * \delta W_{k1l2}) &W_{k1l3} - (lr * \delta W_{k1l3}) \\
W_{k2l1} - (lr* \delta W_{k2l1}) & W_{k2l2} - (lr * \delta W_{k2l2}) &W_{k2l3} - (lr * \delta W_{k2l3}) \\
W_{k3l1} - (lr * \delta W_{k3l1}) & W_{k3l2} - (lr *\delta W_{k3l2}) & W_{k3l3} - (lr * \delta W_{k3l3}) \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\acute{W_{kl}} = \left[ \begin{array}{cccc}
0.1 - (0.01*-0.5531) & 0.4 - (0.01 * -0.0576) & 0.8 - (0.01 * -0.5836) \\
0.3 - (0.01* -0.554347) & 0.7 - (0.01* -0.0577) &0.2 - (0.01* -0.5849) \\
0.5 - (0.01 * -0.577937) & 0.2 - (0.01 -0.06017) & 0.9 - (0.01 * -0.6098) \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\acute{W_{kl}} = \left[ \begin{array}{cccc}
0.105531 & 0.400576 & 0.805836 \\
0.30055 & 0.700577 &0.2005849 \\
0.5005779 & 0.2006017 & 0.9006098 \\ \end{array} \right]
\end{equation*}

#### Finally We made it, Lets jump to the next layer

## BackPropagating the error - (Hidden Layer1 - Hidden Layer2)  Weights

<img src="./images/backprop_layer2.png" width="75%" />

#### Lets calculate a few handy derviatives before we actually calculate the error derviatives wrt Weights in this layer. 

\begin{equation*}
\frac{\partial h2_{out1}}{\partial h2_{in1}} = \frac{\partial Sigmoid(h2_{in1})}{\partial h2_{in1}}
\end{equation*}

\begin{equation*}
\frac{\partial h2_{out1}}{\partial h2_{in1}} = Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h2_{out1}}{\partial h2_{in1}} \\
\frac{\partial h2_{out2}}{\partial h2_{in2}}   \\
\frac{\partial h2_{out3}}{\partial h2_{in3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\
Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\
Sigmoid(h2_{in1}) * (1 - Sigmoid(h2_{in1}))\\ \end{array} \right]
\end{equation*}

#### In our example , this will be 
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h2_{out1}}{\partial h2_{in1}} \\
\frac{\partial h2_{out2}}{\partial h2_{in2}}   \\
\frac{\partial h2_{out3}}{\partial h2_{in3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 Sigmoid(2.73) * (1 - Sigmoid(2.73))\\
Sigmoid(2.76) * (1 - Sigmoid(2.76))\\
Sigmoid(4.001) * (1 - Sigmoid(4.001))\\ \end{array} \right] =  \left[ \begin{array}{cccc}
 0.058156\\
0.0564\\
0.0196\\ \end{array} \right] 
\end{equation*}

#### For each input to neuron lets calculate the derivative with respect to each weight.

##### Now let us look at the final derivative
\begin{equation*}
\frac{\partial h2_{in1}}{\partial W_{j1k1}} = \frac{\partial ((h1_{out1}*W_{j1k1}) + (h1_{out2} * W{j2k1}) + (h1_{out3} * W{j3k1}) + b_{k1})}{\partial W_{j1k1}}
\end{equation*}

#### Now let us look at the final derivative
\begin{equation*}
\frac{\partial h2_{in1}}{\partial W_{j1k1}} = h1_{out1}
\end{equation*}

#### Using similarity we can write:
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h2_{in1}}{\partial W_{j1k1}}  \\
\frac{\partial h2_{in1}}{\partial W_{j2k1}}   \\
\frac{\partial h2_{in1}}{\partial W_{j3k1}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 1.35\\
 1.27\\
 1.8\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h2_{in2}}{\partial W_{j1k2}}  \\
\frac{\partial h2_{in2}}{\partial W_{j2k2}}   \\
\frac{\partial h2_{in2}}{\partial W_{j3k2}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 1.35\\
 1.27\\
 1.8\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h2_{in3}}{\partial W_{j1k3}}  \\
\frac{\partial h2_{in3}}{\partial W_{j2k3}}   \\
\frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 h1_{out1}\\
 h1_{out2}\\
 h1_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 1.35\\
 1.27\\
 1.8\\ \end{array} \right]
\end{equation*}

##### Now we will calulate the change in 
\begin{equation*}
W_{j3k1}
\end{equation*} 
##### and generalize it for all other variables.

#### Caution: Make sure that you have understood everything we discussed till here. 

##### This will be simply
\begin{equation*}
\frac{\partial E_{total}}{\partial W_{j3k1}}
\end{equation*}

##### Using chain rule:
\begin{equation*}
\frac{\partial E_{total}}{\partial W_{j3k1}} = \frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2
_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial W_{j3k1}}
\end{equation*}

##### Now we will see each and every equation individually.


#### Lets look at the matrix
### By symmetry
\begin{equation*}
\delta W_{jk} = \left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial W_{j1k2}} & \frac{\partial E_{total}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial W_{j2k1}} & \frac{\partial E_{total}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial W_{j3k3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
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
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{array} \right]
\end{equation*}

#### We have Already calculated the 2nd and 3rd term in each matrix. We need to check on the 1st term. If we see the matrix, the first term is common in all the columns. So there are only three values. Lets look into one value

\begin{equation*}
\frac{\partial E_{total}}{\partial h2_{out1}} = \frac{\partial E_{1}}{\partial h2_{out1}} + \frac{\partial E_{2}}{\partial h2_{out1}} + \frac{\partial E_{3}}{\partial h2_{out1}}
\end{equation*}

#### Lets see what each individual term boils down too.

\begin{equation*}
\frac{\partial E_{1}}{\partial h2_{out1}} = \frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out1}}
\end{equation*}

\begin{equation*}
\frac{\partial E_{2}}{\partial h2_{out1}} = \frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out1}}
\end{equation*}

\begin{equation*}
\frac{\partial E_{3}}{\partial h2_{out1}} = \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out1}}
\end{equation*}

### BY symmentry

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out1}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out1}}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out1}})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out2}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out2}}) + (\frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out2}})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * \frac{\partial O_{in1}}{\partial h2_{out3}}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * \frac{\partial O_{in2}}{\partial h2_{out3}}) + (\frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * \frac{\partial O_{in3}}{\partial h2_{out3}})\\ \end{array} \right]
\end{equation*}

__Again the first two values are already calculated by us when dealing with derviatives of W_{kl}. We just need to calculate the third one, Which is the derivative of  input to each output layer wrt output of hidden layer-2. It is nothing but the corresponding weight which connects both the layers.__

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial O_{inl}}{\partial h2_{out1}} & \frac{\partial O_{in2}}{\partial h2_{out1}} & \frac{\partial O_{in3}}{\partial h2_{out1}} \\
\frac{\partial O_{inl}}{\partial h2_{out2}} & \frac{\partial O_{in2}}{\partial h2_{out2}} & \frac{\partial O_{in3}}{\partial h2_{out2}}\\
\frac{\partial O_{inl}}{\partial h2_{out3}} & \frac{\partial O_{in2}}{\partial h2_{out3}} & \frac{\partial O_{in3}}{\partial h2_{out3}} \\ \end{array} \right] = \left[ \begin{array}{cccc}
W_{k1l1} & W_{k1l2} & W_{k1l3} \\
W_{k2l1} & W_{k2l2} & W_{k2l3} \\
W_{k3l1} & W_{k3l2} & W_{k3l3} \\ \end{array} \right] 
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k1l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k1l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k1l3})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k2l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k2l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k2l3})\\
(\frac{\partial E_{1}}{\partial O_{out1}} * \frac{\partial O_{out1}}{\partial O_{in1}} * W_{k3l1}) + (\frac{\partial E_{2}}{\partial O_{out2}} * \frac{\partial O_{out2}}{\partial O_{in2}} * W_{k3l2}) + ( \frac{\partial E_{3}}{\partial O_{out3}} * \frac{\partial O_{out3}}{\partial O_{in3}} * W_{k2l3})\\
\end{array} \right]
\end{equation*}

#### All Values are calculated before we just need to impute the corresponding values for our example.
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial h2_{out1}}  \\
\frac{\partial E_{total}}{\partial h2_{out2}}   \\
\frac{\partial E_{total}}{\partial h2_{out3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
(-3.70644 *0.15911 * 0.1) + (-1.4755 * 0.2040 *0.4) + ( -1.6886 * 0.3685 * 0.8)\\
(-3.70644 * 0.15911 * 0.3) + (-1.4755 * 0.2040 * 0.7) + ( -1.6886 * 0.3685 * 0.2)\\
(-3.70644 * 0.15911  * 0.5) + (-1.4755 * 0.2040 * 0.2) + ( -1.6886 * 0.3685 * 0.9)\\
\end{array} \right] =  \left[ \begin{array}{cccc}
(-0.0589) + (-0.2383) + ( -0.5931)\\
(-0.1769) + (-0.417) + ( -0.14828)\\
(-0.2948) + (-0.119) + ( -0.667)\\
\end{array} \right] =
\left[ \begin{array}{cccc}
-0.8903\\
-0.74218\\
-1.0810\\
\end{array} \right]
\end{equation*}

#### Lets look at the matrix
### By symmetry
\begin{equation*}
\delta W_{jk} = \left[ \begin{array}{cccc}
\frac{\partial E_{total}}{\partial W_{j1k1}} & \frac{\partial E_{total}}{\partial W_{j1k2}} & \frac{\partial E_{total}}{\partial W_{j1k3}} \\
\frac{\partial E_{total}}{\partial W_{j2k1}} & \frac{\partial E_{total}}{\partial W_{j2k2}} & \frac{\partial E_{total}}{\partial W_{j2k3}} \\
\frac{\partial E_{total}}{\partial W_{j3k1}} & \frac{\partial E_{total}}{\partial W_{j3k2}} & \frac{\partial E_{total}}{\partial W_{j3k3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
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
_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial W_{j3k3}} \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\delta W_{jk} =  \left[ \begin{array}{cccc}
-0.8903 * 0.058156 * 1.35 & -0.74218 * 0.0564 * 1.35 & -1.0810 * 0.0196 * 1.35 \\
-0.8903 * 0.058156  * 1.27 & -0.74218 * 0.0564 * 1.27 & -1.0810 * 0.0196 * 1.27 \\
-0.8903 * 0.058156 * 1.8 & -0.74218 * 0.0564 * 1.8 & -1.0810 * 0.0196 * 1.8 \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\delta W_{jk} =  \left[ \begin{array}{cccc}
-0.06989 & -0.0565 & -0.0286 \\
-0.06575 & -0.05316 & -0.0269 \\
-0.0932 & -0.0753 & -0.03813 \\ \end{array} \right]
\end{equation*}

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
\begin{equation*}
\acute{W_{jk}} = \left[ \begin{array}{cccc}
W_{j1k1} - (lr*\delta W_{j1k1}) & W_{j1k2} - (lr * \delta W_{j1k2}) &W_{j1k3} - (lr * \delta W_{j1k3}) \\
W_{j2k1} - (lr* \delta W_{j2k1}) & W_{j2k2} - (lr * \delta W_{j2k2}) &W_{j2k3} - (lr * \delta W_{j2k3}) \\
W_{j3k1} - (lr * \delta W_{j3k1}) & W_{j3k2} - (lr *\delta W_{j3k2}) & W_{j3k3} - (lr * \delta W_{j3k3}) \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\acute{W_{jk}} = \left[ \begin{array}{cccc}
0.2 - (0.01*-0.06989) & 0.3 - (0.01 * -0.0565) & 0.5 - (0.01 * -0.0286) \\
0.3 - (0.01* -0.06575) & 0.5 - (0.01* -0.05316) &0.7 - (0.01* -0.0269) \\
0.6 - (0.01 * -0.0932) & 0.4 - (0.01 -0.0753) & 0.8 - (0.01 * -0.03813) \\ \end{array} \right]
\end{equation*}

\begin{equation*}
\acute{W_{jk}} = \left[ \begin{array}{cccc}
0.2006989 & 0.300565 & 0.500286 \\
0.3006575 & 0.5005316 &0.700269 \\
0.600932 & 0.400753 & 0.803813 \\ \end{array} \right]
\end{equation*}

#### Finally We made it, Lets jump to the next layer

## BackPropagating the error - (Input Layer - Hidden Layer1)  Weights

<img src="./images/backprop_layer1.png" width="75%" />

#### Lets calculate a few handy derviatives before we actually calculate the error derviatives wrt Weights in this layer. 

\begin{equation*}
\frac{\partial h1_{out1}}{\partial h1_{in1}} = \frac{\partial Relu(h1_{in1})}{\partial h1_{in1}}
\end{equation*}

#### We already know 
### Relu
\begin{equation*}
relu = max(0, x) 
\end{equation*}

\begin{equation*}
\ if x >0 , \frac{\partial (relu)}{\partial x} = 1
\end{equation*}

\begin{equation*}
\ Otherwise , \frac{\partial (relu)}{\partial x} = 0
\end{equation*}

### Since the inputs are positive

\begin{equation*}
\frac{\partial h1_{out1}}{\partial h1_{in1}} = 1.0
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h1_{out1}}{\partial h1_{in1}} \\
\frac{\partial h1_{out2}}{\partial h1_{in2}}   \\
\frac{\partial h1_{out3}}{\partial h1_{in3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 1.0\\
1.0\\
1.0\\ \end{array} \right]
\end{equation*}


#### For each input to neuron lets calculate the derivative with respect to each weight.

#### Now let us look at the final derivative
\begin{equation*}
\frac{\partial h1_{in1}}{\partial W_{i1j1}} = \frac{\partial ((I_{out1}*W_{j1k1}) + (I_{out2} * W{i2j1}) + (I_{out3} * W{i3j1}) + b_{j1})}{\partial W_{i1j1}}
\end{equation*}

#### Now let us look at the final derivative
\begin{equation*}
\frac{\partial h1_{in1}}{\partial W_{i1j1}} = I_{out1}
\end{equation*}

#### Using similarity we can write:
\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h1_{in1}}{\partial W_{i1j1}}  \\
\frac{\partial h1_{in1}}{\partial W_{i2j1}}   \\
\frac{\partial h1_{in1}}{\partial W_{i3j1}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.1\\
 0.2\\
 0.7\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h1_{in2}}{\partial W_{i1j2}}  \\
\frac{\partial h1_{in2}}{\partial W_{i2j2}}   \\
\frac{\partial h1_{in2}}{\partial W_{i3j2}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.1\\
 0.2\\
 0.7\\ \end{array} \right]
\end{equation*}

\begin{equation*}
\left[ \begin{array}{cccc}
\frac{\partial h1_{in3}}{\partial W_{i1j3}}  \\
\frac{\partial h1_{in3}}{\partial W_{i2j3}}   \\
\frac{\partial h1_{in3}}{\partial W_{i3j3}} \\ \end{array} \right] =  \left[ \begin{array}{cccc}
 I_{out1}\\
 I_{out2}\\
 I_{out3}\\ \end{array} \right] = \left[ \begin{array}{cccc}
 0.1\\
 0.2\\
 0.7\\ \end{array} \right]
\end{equation*}

##### Now we will calulate the change in 
$$
W_{j3k1}
$$ 

##### and generalize it for all other variables.

#### Caution: Make sure that you have understood everything we discussed till here. 

##### This will be simply
$$
\frac{\partial E_{total}}{\partial W_{i2j1}}
$$

##### Using chain rule:
$$
\frac{\partial E_{total}}{\partial W_{i2j1}} = \frac{\partial E_{total}}{\partial h1_{out1}} * \frac{\partial h1
_{out1}}{\partial h1_{in1}} * \frac{\partial h1_{in1}}{\partial W_{i2j1}}
$$

##### Now we will see each and every equation individually.

#### Lets look at the matrix
### By symmetry
$$
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
$$

#### We know the 2nd and 3rd derivatives in each cell in the above matrix. Lets look at how to get to derivative of 1st term in each cell.

$$
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * \frac{\partial h2_{in1}}{\partial h1_{out1}}) \\
(\frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2_{out2}}{\partial h2_{in2}} * \frac{\partial h2_{in2}}{\partial h1_{out2}})\\
(\frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2_{out3}}{\partial h2_{in3}} * \frac{\partial h2_{in3}}{\partial h1_{out3}})\\ \end{bmatrix} 
$$

### We have calculated all the values previously except the last one in each cell, which is a simple derivative of linear terms.

$$
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
(\frac{\partial E_{total}}{\partial h2_{out1}} * \frac{\partial h2_{out1}}{\partial h2_{in1}} * W_{j1k1} \\
(\frac{\partial E_{total}}{\partial h2_{out2}} * \frac{\partial h2_{out2}}{\partial h2_{in2}} * W_{j2k2}\\
(\frac{\partial E_{total}}{\partial h2_{out3}} * \frac{\partial h2_{out3}}{\partial h2_{in3}} * W_{j3k3}\\ \end{bmatrix} 
$$

$$
 \begin{bmatrix}
\frac{\partial E_{total}}{\partial h1_{out1}}  \\
\frac{\partial E_{total}}{\partial h1_{out2}}   \\
\frac{\partial E_{total}}{\partial h1_{out3}} \\ \end{bmatrix}  =   \begin{bmatrix}
-0.8903 *0.058156 * 0.2 \\
-0.74218 * 0.0564 *0.5\\
-1.0810 * 0.0196 * 0.8\\ \end{bmatrix}  =   \begin{bmatrix}
-0.01035 \\
-0.0209\\
-0.0169\\ \end{bmatrix} 
$$

#### Lets look at the matrix
### By symmetry
$$
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
$$

$$
\delta W_{ij} =   \begin{bmatrix}
-0.01035 * 1 * 0.1 & -0.0209 * 1 * 0.1 & -0.0169 * 1 * 0.1 \\
-0.01035 * 1  * 0.2 & -0.0209* 1 * 0.2 & -0.0169 * 1 * 0.2 \\
-0.01035 * 1 * 0.7& -0.0209 * 1 * 0.7 & -0.0169 * 1 * 0.7 \\ \end{bmatrix} 
$$

$$
\delta W_{ij} =   \begin{bmatrix}
-0.001035 & -0.00209 & -0.00169 \\
-0.00207 & -0.00418 & -0.00338 \\
-0.007245 & -0.01463 & -0.01183 \\ \end{bmatrix} 
$$

#### Consider a learning rate (lr) of 0.01 We get our final Weight matrix as 
$$
\acute{W_{ij}} =  \begin{bmatrix}
W_{i1j1} - (lr*\delta W_{i1j1}) & W_{i1j2} - (lr * \delta W_{i1j2}) &W_{i1j3} - (lr * \delta W_{i1j3}) \\
W_{i2j1} - (lr* \delta W_{i2j1}) & W_{i2j2} - (lr * \delta W_{i2j2}) &W_{i2j3} - (lr * \delta W_{i2j3}) \\
W_{i3j1} - (lr * \delta W_{i3j1}) & W_{i3j2} - (lr *\delta W_{i3j2}) & W_{i3j3} - (lr * \delta W_{i3j3}) \\ \end{bmatrix} 
$$

$$
\acute{W_{ij}} =  \begin{bmatrix}
0.1 - (0.01*-0.001035) & 0.2 - (0.01 * -0.00209) & 0.3 - (0.01 * -0.00169) \\
0.3 - (0.01* -0.00207) & 0.2 - (0.01* -0.00418) &0.7 - (0.01* -0.00338) \\
0.4 - (0.01 * -0.0007245) & 0.3 - (0.01 -0.01463) & 0.9 - (0.01 * -0.01183) \\ \end{bmatrix} 
$$

$$
\acute{W_{ij}} =  \begin{bmatrix}
0.10001035 & 0.2000209 & 0.3000169 \\
0.3000207 & 0.2000418 &0.7000338 \\
0.40007245 & 0.3001463 & 0.9001183 \\ \end{bmatrix} 
$$

#### The End 

## Our Inital Weights 


$$
W_{ij} =  \begin{bmatrix}
W_{i1j1} & W_{i1j2} & W_{i1j3} \\
W_{i2j1} & W_{i2j2} & W_{i2j3} \\
W_{i3j1} & W_{i3j2} & W_{i3j3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.1 & 0.2 & 0.3 \\
0.3 & 0.2 & 0.7 \\
0.4 & 0.3 & 0.9 \\ \end{bmatrix} 
$$


$$
W_{jk} =  \begin{bmatrix}
W_{j1k1} & W_{j1k2} & W_{j1k3} \\
W_{j2k1} & W_{j2k2} & W_{j2k3} \\
W_{j3k1} & W_{j3k2} & W_{j3k3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.2 & 0.3 & 0.5 \\
0.3 & 0.5 & 0.7 \\
0.6 & 0.4 & 0.8 \\ \end{bmatrix} 
$$


$$
W_{kl} =  \begin{bmatrix}
W_{k1l1} & W_{k1l2} & W_{k1l3} \\
W_{k2l1} & W_{k2l2} & W_{k2l3} \\
W_{k3l1} & W_{k3l2} & W_{k3l3} \\ \end{bmatrix}  =   \begin{bmatrix}
0.1 & 0.4 & 0.8 \\
0.3 & 0.7 & 0.2 \\
0.5 & 0.2 & 0.9 \\ \end{bmatrix} 
$$

## Our final weights 
$$
\acute{W_{ij}} =  \begin{bmatrix}
0.10001035 & 0.2000209 & 0.3000169 \\
0.3000207 & 0.2000418 &0.7000338 \\
0.40007245 & 0.3001463 & 0.9001183 \\ \end{bmatrix} 
$$

$$
\acute{W_{jk}} =  \begin{bmatrix}
0.2006989 & 0.300565 & 0.500286 \\
0.3006575 & 0.5005316 &0.700269 \\
0.600932 & 0.400753 & 0.803813 \\ \end{bmatrix} 
$$

$$
\acute{W_{kl}} =  \begin{bmatrix}
0.105531 & 0.400576 & 0.805836 \\
0.30055 & 0.700577 &0.2005849 \\
0.5005779 & 0.2006017 & 0.9006098 \\ \end{bmatrix} 
$$

## Important Notes:
- I have completely eliminated bias when differentiating. Do you know why ?
- Backprop of bias should be straight forward. Try on your own.
- I have taken only one example. What will happen if we take batch of examples?
- Though I have not mentioned directly about vansihing gradients. Do you see why it occurs?
- What would happen if all the weights are the same number instead of random ?
