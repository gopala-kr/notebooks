# Keras-FlappyBird

A single 200 lines of python code of tutorial DQN with Keras

![](animation1.gif)

Based on the code from: 

https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

# Installation Dependencies:

* Python 2.7
* Keras 1.0 
* pygame
* scikit-image

# How to Run?

**CPU only**

```
git clone https://github.com/ypeleg/KerasRLTutorial
cd Keras-FlappyBird
python qlearn.py 
```

**GPU version (Theano)**

```
git clone https://github.com/ypeleg/KerasRLTutorial
cd Keras-FlappyBird
THEANO_FLAGS=device=gpu,floatX=float32,lib.cnmem=0.2 python qlearn.py 
```

If you want to train the network from beginning, delete the model.h5 and run qlearn.py with the mode=train
