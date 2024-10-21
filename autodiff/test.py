import numpy as np
import yaml
from activation import ReLU, Sigmoid, Linear
from network import Network, NetworkParams, PredictionType
from utils import plot_results, create_network, train_network
import pickle as pkl




if __name__ == '__main__':
    file_path = 'assignment-one-test-parameters.pkl'

    with open(file_path, 'rb') as file:
        data = pkl.load(file)

    config_path = '/Users/ciceku/Desktop/untitled folder/autodiff/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    network = create_network(data, config)
    
    loss_vals = train_network(network, data, config)
    plot_results(loss_vals,config)
    


   