import numpy as np
import yaml
from autodiff.activation import ReLU, Sigmoid, Linear
from autodiff.network import Network, NetworkParams
from utils import plot_results, create_network, train_network, train_pytorch_network, plot_results_torch
from autodiff.config import CONFIG_PATH, DATA_PATH
import pickle as pkl




if __name__ == '__main__':
    file_path = DATA_PATH

    with open(file_path, 'rb') as file:
        data = pkl.load(file)

    config_path = CONFIG_PATH
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    network = create_network(data, config)
    
    loss_vals = train_network(network, data, config)
    loss_vals_torch = train_pytorch_network(data, config)
    plot_results(loss_vals,config)
    plot_results_torch(loss_vals_torch, config)
    


   