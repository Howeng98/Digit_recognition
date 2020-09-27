# Handwriting Digit Recognition with Convolutional Neural Network
Objective of this work is to identify and classify the Handwriting Digit Images from group 0 to 9 with ```Convolutional Neural Network``` 
And the model predict accuracy is **0.9918**. In addiction, ```google colaboratory GPU``` is recommended for the model training if you can't afford a better Graphic Card



## Structure
1. **Neural Network Structure**

![](img/neural_network.png)

2. **Model Structure Sample**

![](img/cnn_structure.jpeg)

3. **Model Structure and Layes**

![.](img/model.png)


## Requirement
  - **Python 3.8.2 or above**
  - **GPU (recommended)**
  - **Tensorflow**
  - **Keras**
  - **Matplotlib**
  - **MNIST dataset**
  
## Build
```
python3 digit_recognition.py
```
## Output 
It show the handwriting image which provided by the MNIST and the model predict result 

![](img/output1.png)
![](img/output2.png)
![](img/output3.png)

## Prework and setup
As above mentioned, if you are using google colaboratory GPU then you must install following module and packages for compile and build

  - **python-mnist**
  - **keras**
  
```
!pip install python-mnist
!pip install keras
```
