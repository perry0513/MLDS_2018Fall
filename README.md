# MLDS_hw1
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installing](#installing)
* [HW1-1](#HW1-1)
    * [Function Simulation](#function-simulation)
        * [Models](#dnn-models)
        * [Simulated Function](#simulated-functions)
        * [Conclusion](#dnn-conclusion)
    * [Train on Actual Task](#train-on-actual-task)
        * [CNN Models](#cnn-models)
        * [Loss and Accuracy](#loss-and-accuracy)
        * [Conclusion](#cnn-conclusion)
* [HW1-2](#HW1-2)
    * [Visualize the optimization process](#visualize-the-optimization-process)
    * [Observe gradient norm during training](#observe-gradient-norm-during-training)
        * [Conclusion](#observe-conclusion)
    * [What happens when gradient is almost zero](#what-happens-when-gradient-is-almost-zero)
        * [Conclusion](#zero-conclusion)
    * [Visualize the error surface](#visualize-the-error-surface)

## Getting Started
### Prerequisites
    $ pip install tensorflow
    $ pip install keras
#### Pytorch
    go to https://pytorch.org/
    choose your operting system and install pytorch
### Installing
#### [Function Simulation](##function-simulation)
##### 1 for the first function, 2 for the second
    $ python sim_func.py 1
    $ python sim_func.py 2
#### [Train on Actual Task](##train-on-actual-task)
    $ python cnn.py
#### [Visualize the optimization process](#visualize-the-optimization-process)
    $ python pca.py
#### [What happens when gradient is almost zero](#what-happens-when-gradient-is-almost-zero)
    $ python min_ratio.py

## HW1-1    
### Function Simulation
<a id="dnn-models"></a>

#### Models
![](readme_src/hw1-1/sim_model1,2.png)
![](readme_src/hw1-1/sim_model3,4.png)

#### Simulated Functions
- y = sin(3πx) + sin(4πx)

    ![](readme_src/hw1-1/sim_function1.png)
- y = esp(sin(40x)) * log(x+1)

    ![](readme_src/hw1-1/sim_function2.png)
<a id="dnn-conclusion"></a>

#### Conclusion 
    Deeper is better (in our case)
    Functions cannot be too simple or complicated
    LeakyReLU seems to work better than ReLU

### Train on Actual Task
<a id="cnn-models"></a>

#### Models
![](readme_src/hw1-1/cnn_model1,2.png)
![](readme_src/hw1-1/cnn_model3,4.png)
#### Loss and Accuracy
![](readme_src/hw1-1/cnn_loss&accuracy.png)
<a id="cnn-conclusion"></a>

#### Conclusion 
    Thicker convolutional layer may not perform better
    For such easy cases, more hidden layers seem not to be beneficial
    Model 4 performs the best

## HW1-2
### Visualize the optimization process
![](readme_src/hw1-2/visualize.png)
### Observe gradient norm during training
![](readme_src/hw1-2/observe-gradient.png)
<a id="observe-conclusion"></a>

#### Conclusion
    Norm of gradient lowers together obviously.
    However when it comes to complicated functions, such as exp(.sin(40*x))*log(x+1), it is hard to master gradient norm's behavior. (but it still tends to decrease)
### What happens when gradient is almost zero
![](readme_src/hw1-2/zero-gradient_1.png)
![](readme_src/hw1-2/zero-gradient_2.png)
<a id="zero-conclusion"></a>

#### Conclusion
    When we  start training on gradient norm, there will be a jump on both loss and norm.
    When train with complicate function, result can get worse when minimizing gradient norm.
### Visualize the error surface