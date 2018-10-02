# MLDS_hw1
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installing](#installing)
* [Function Simulation](#function-simulation)
    * [Models](#dnn-models)
    * [Simulated Function](#simulated-functions)
    * [Conclusion](#dnn-conclusion)
* [Train on Actual Task](#train-on-actual-task)
    * [CNN Models](#cnn-models)
    * [Loss and Accuracy](#loss-and-accuracy)
    * [Conclusion](#cnn-conclusion)
## Getting Started
### Prerequisites
    $ pip install tensorflow
    $ pip install keras
### Installing
#### [Function Simulation](##function-simulation)
##### 1 for the first function, 2 for the second
    $ python sim_func.py 1
    $ python sim_func.py 2
#### [Train on Actual Task](##train-on-actual-task)
    $ python cnn.py
## Function Simulation
<a id="dnn-models"></a>

### Models
![](readme_src/hw1-1/sim_model1,2.png)
![](readme_src/hw1-1/sim_model3,4.png)

### Simulated Functions
- y = sin(3πx) + sin(4πx)

    ![](readme_src/sim_function1.png)
- y = esp(sin(40x)) * log(x+1)

    ![](readme_src/sim_function2.png)
<a id="dnn-conclusion"></a>

### Conclusion 
    Deeper is better (in our case)
    Functions cannot be too simple or complicated
    LeakyReLU seems to work better than ReLU

## Train on Actual Task
<a id="cnn-models"></a>

### Models
![](readme_src/hw1-1/cnn_model1,2.png)
![](readme_src/hw1-1/cnn_model3,4.png)
### Loss and Accuracy
![](readme_src/hw1-1/cnn_loss&accuracy.png)
<a id="cnn-conclusion"></a>

### Conclusion 
    Thicker convolutional layer may not perform better
    For such easy cases, more hidden layers seem not to be beneficial
    Model 4 performs the best

