import numpy as np
from typing import TypedDict

from .activation import Activation, get_activation
from .config import GRAD_DTYPE, WEIGHT_DTYPE



class LayerParams(TypedDict):
    """Dict structure defining the required information to initialize a ff layer"""
    input_shape: int
    n_neurons: int
    weight_init: np.ndarray
    bias_init: np.ndarray
    activation: str

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
    activations: np.ndarray
    activation_func: Activation

    def __init__(self, params: LayerParams): 
        self.input_shape = params["input_shape"]
        self.n_neurons = params["n_neurons"]

        # TODO: check that weights are valid shape
        self.w = np.array(params["weights_init"], dtype=WEIGHT_DTYPE)
        # TODO: check that biases are valid shape
        self.b = np.array(params["bias_init"], dtype=WEIGHT_DTYPE)

        self.activation_func = get_activation(params["activation"])

        # Neuron activations and parameter gradients 
        self.w_grads = np.zeros_like(self.w, dtype=GRAD_DTYPE)
        self.b_grads = np.zeros_like(self.b, dtype=GRAD_DTYPE)
        self.activations = np.zeros(self.n_neurons, dtype=WEIGHT_DTYPE)

    def forward(self, input: np.array): 
        """Propagates the input forward and records activations of each neuron

        Args:
            input (np.array): inputs to the neurons (M, 1)

        Returns: 
            output (np.array): outputs from the neurons

        A = W @ x + b
        """
        # TODO: verify shapes
        self.activations = self.w @ input + self.b
        return self.activation_func.activate(self.activations)
    

    def backward(self, delta: np.array):
        """Propagates loss (delta) backward and adds to the gradients of each param
        
        Args: 
            delta: the errors from the proceeding layer
            
        Returns: 
            delta_pre: the errors to be passed back
        """
        g_prime = self.activation_func.derivate(self.activations)
        delta = g_prime * self.w.T @ delta
        
        
        pass

    
    def zero_grad(self): 
        """zeros the gradients"""
        self.w_grads.fill(0.0)
        self.b_grads.fill(0.0)
        return
    

    def update_params(self, learning_rate: float): 
        """Perform a weight update with the given learning rate

        Args:
            learning_rate (float): the amount to update the weights

            w_updated = w - learning_rate * gradient
        """
    

    

    