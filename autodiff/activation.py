import numpy as np
from enum import Enum
from abc import ABC, abstractmethod


class ActivationFunc(Enum):
    LINEAR = "linear"
    RELU = "relu"
    SIGMOID = "sigmoid"


class Activation(ABC):
    @abstractmethod
    def activate(self, a: np.ndarray) -> np.ndarray:
        """Implement g(a)"""
        pass

    @abstractmethod
    def derivate(self, a: np.ndarray) -> np.ndarray:
        """Implement g'(a)"""
        pass


class Linear(Activation):
    def activate(self, a: np.ndarray) -> np.ndarray:
        return a 
    
    def derivate(self, a: np.ndarray) -> np.ndarray: 
        return np.ones_like(a)


class ReLU(Activation): 
    def activate(self, a: np.ndarray) -> np.ndarray:
        a[a<0] = 0
        return a
    
    def derivate(self, a: np.ndarray) -> np.ndarray: 
        derivative = np.ones_like(a)
        derivative[a<0] = 0
        return derivative
    

class Sigmoid(Activation):
    def activate(self, a: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-a))
    
    def derivate(self, a: np.ndarray) -> np.ndarray: 
        activate = self.activate(a)
        return activate * (1-activate)