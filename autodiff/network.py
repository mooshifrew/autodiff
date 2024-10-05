import numpy as np
from enum import Enum
from typing import TypedDict, List
from .layer import LayerParams, Layer
from .activation import Activation, Linear, ReLU, Sigmoid

class PredictionType(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class NetworkParams(TypedDict):
    """Dict structure defining the required information to initialize a ff network"""
    input_shape: int
    output_shape: int
    prediction_type: PredictionType # regression | classification
    layers: List[LayerParams]


class Network: 
    """Simple fully connected feed-forward neural network with backpropagation"""

    input_shape: int
    output_shape: int
    output_activation: Activation
    layers: List[Layer]
    
    def __init__(self, params: NetworkParams):
        self.input_shape = params["input_shape"]
        self.output_shape = params["output_shape"]
        self.layers = []
        for layer_param in params["layers"]: 
            self.layers.append(Layer(layer_param))

        if params["prediction_type"] == PredictionType.REGRESSION: 
            self.output_activation = Linear()
        else: 
            self.output_activation = Sigmoid()


    def forward(self, x: np.array):
        """Propagates an input through the network and stores activations

        Args:
            x (np.array): input array
        """
        for layer in self.layers: 
            x = layer.forward(x)
            print(x)
        
        return x

    def backward(self, delta: np.array):
        """Backpropagates loss through the network

        Args:
            delta (np.array): the loss to propagate
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return
    
    def update_params(self):
        """Update weights in each layer"""
        for layer in self.layers: 
            layer.update_params()
        return
        
    def zero_grad(self): 
        """Set all gradients to zero"""
        for layer in self.layers: 
            layer.zero_grad()




