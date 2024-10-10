import numpy as np
from typing import TypedDict

from .activation import Activation
from .config import GRAD_DTYPE, WEIGHT_DTYPE


class LayerParams(TypedDict):
    """Dict structure defining the required information to initialize a ff layer"""
    input_shape: int
    n_neurons: int
    weight_init: np.ndarray
    bias_init: np.ndarray
    activation: Activation

class Layer: 
    """Feedforward layer with support for auto-differentiation
    
    If N is the number of neurons, and M is the number of inputs, weights are of 
    dimension (N, M)
    """
    input_shape: int
    n_neurons: int
    w: np.ndarray
    b: np.ndarray
    w_grads: np.ndarray
    b_grads: np.ndarray
    activations: np.ndarray # A = W @ x + b
    last_input: np.ndarray # (M, 1)
    activation_func: Activation

    def __init__(self, params: LayerParams): 
        self.input_shape = params["input_shape"]
        self.n_neurons = params["n_neurons"]

        # TODO: check that weights are valid shape
        self.w = np.array(params["weight_init"], dtype=WEIGHT_DTYPE)
        # TODO: check that biases are valid shape
        self.b = np.array(params["bias_init"], dtype=WEIGHT_DTYPE)

        assert self.w.shape == (self.n_neurons, self.input_shape), "weights wrong dim"
        assert self.b.shape == (self.n_neurons,), "biases wrong dim"

        self.activation_func = params["activation"]

        # Neuron activations and parameter gradients 
        self.w_grads = np.zeros_like(self.w, dtype=GRAD_DTYPE)
        self.b_grads = np.zeros_like(self.b, dtype=GRAD_DTYPE)
        self.activations = np.zeros(self.n_neurons, dtype=WEIGHT_DTYPE)
        self.last_input = np.zeros(self.input_shape, dtype=WEIGHT_DTYPE)

    def forward(self, input: np.ndarray): 
        """Propagates the input forward and records activations of each neuron

        Args:
            input (np.array): inputs to the neurons (M,)

        Returns: 
            output (np.array): outputs from the neurons

        A = W @ x + b
        """
        # TODO: verify shapes
        self.last_input = input
        self.activations = self.w @ input + self.b
        return self.activation_func.activate(self.activations)
    

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Propagates loss (delta) backward and adds to the gradients of each param
        
        Args: 
            delta: the errors
            
        Returns: 
            delta_pre: the errors to pass back

        dL/dw = dL/dz  dz/da  da/dw (N,M)
        dL/dz = delta (N,)
        dz/da = g_prime (N,)
        da/dw = last_input (M,)
        """

        g_prime = self.activation_func.derivate(self.activations)
        w_grad_update = np.outer(delta * g_prime, self.last_input)
        b_grad_update = delta * g_prime # treat this like another weight where the input is always 1
        self.w_grads += w_grad_update
        self.b_grads += b_grad_update
        print(self.w_grads)

        # get the delta to pass back
        delta_pre = self.w.T @ (g_prime * delta)
        return delta_pre

    
    def zero_grad(self): 
        """zeros the gradients"""
        self.w_grads.fill(0.0)
        self.b_grads.fill(0.0)
        return
    

    def update_params(self, learning_rate: float, batch_size: int): 
        self.w -= learning_rate * (self.w_grads / batch_size)
        self.b -= learning_rate * (self.b_grads / batch_size)

    
    def print_params(self):
        print(f"Neurons:    {self.n_neurons}; Inputs: {self.input_shape}")
        print(f"Weights:{self.w.shape} {self.w}")
        print(f"Biases:{self.b.shape}  {self.b}")
        print(f"W grads: {self.w_grads.shape} {self.w_grads}")
        print(f"B grads: {self.b_grads.shape}   {self.b_grads}")
        print(f"Activation: {self.activations}")
        print(f"Last input: {self.last_input}")
        return
