import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sine(x):
    return np.sin(x)

def cosine(x):
    return np.cos(x)

def softmax(x):
    if isinstance(x, float):
        raise TypeError("Expected array-like object, got float")
    return np.exp(x) / np.sum(np.exp(x))

activations = {
    "sigmoid": sigmoid,
    "relu": relu,
    "tanh": tanh,
    "sine": sine,
    "cosine": cosine,
    "softmax": softmax
}
