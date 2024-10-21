import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl


from autodiff.activation import ReLU, Sigmoid, Linear, Activation
from autodiff.network import Network, NetworkParams
from model import SimpleNet, set_model_weights
from autodiff.config import CONFIG_PATH, DATA_PATH

def loss_fn(y, y_hat): 
    return 0.5*((y_hat - y)**2)

def create_network(data: dict, config: dict):
    input_shape:int = config['NETWORK']['INPUT_SHAPE']
    output_shape:int  = config['NETWORK']['OUTPUT_SHAPE']
    activation_name: str = config['NETWORK']['ACTIVATION']
    is_regression: bool = config['NETWORK']['IS_REGRESSION']

    if activation_name.lower() == 'relu':
        activation: Activation =  ReLU()
    elif activation_name.lower() == 'linear':
        activation: Activation =  Linear()
    elif activation_name.lower() == 'sigmoid':
        activation: Activation =  Sigmoid()
    else:
        raise ('Enter a valid activation function')

    if is_regression == True:
        classifaction_type: Activation = Linear()
    else:
        classifaction_type: Activation = Sigmoid()

    network_def: NetworkParams = {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "layers": [
            {
                "input_shape": 2,
                "n_neurons": len(data['w1']),
                "weight_init": data['w1'],
                "bias_init": data['b1'] ,
                "activation": activation,
            },
            {
                "input_shape": len(data['w1']),
                "n_neurons": len(data['w2']),
                "weight_init": data['w2'],
                "bias_init": data['b2'] ,
                "activation": activation,
            },
            {
                "input_shape": len(data['w2']),
                "n_neurons": len(data['w3']),
                "weight_init": data['w3'],
                "bias_init": data['b3'] ,
                "activation": classifaction_type, 
            }
        ]
    } 
    network = Network(network_def)

    return network
    

def plot_results(loss_vals: list, config: dict):
    epochs = config['TRAIN']['EPOCH']
    print(loss_vals)

    plt.plot(range(epochs), [elem.detach().numpy() for elem in loss_vals])
    plt.title('Pytorch Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.show()
    

def train_network( network: Network, data: dict,config: dict):
    epoch = config['TRAIN']['EPOCH']
    lr = config['TRAIN']['LR']
    
    loss_vals = []

    for iteration in range(epoch): 

        current_loss = 0
        network.zero_grad()
        for index, input in enumerate(data['inputs']):
            y_pred = network.forward(input)
            y = data['targets'][index] 
            loss = loss_fn(y, y_pred)
            current_loss += loss
            
            network.backward(y_pred - y)

        print(f'Epoch {iteration+1}/{epoch}, Loss: {(current_loss/len(data["inputs"]))[0]}')
        network.update_params(lr, len(data['inputs']))
        loss_vals.append(current_loss/len(data["inputs"]))
    return loss_vals


def train_pytorch_network(data: dict, config: dict):
    epochs = config['TRAIN']['EPOCH']
    lr = config['TRAIN']['LR']
    
    loss_vals = []

    inputs = torch.tensor(data['inputs'], dtype=torch.float64)
    targets = torch.tensor(data['targets'], dtype=torch.float64)

    model = SimpleNet()
    set_model_weights(data, model)

    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs)
        loss = criterion(output.squeeze(), targets)/2

        # Accumulate gradients
        loss.backward()

        # Perform a single parameter update after accumulating gradients
        optimizer.step()

        loss_vals.append(loss)

        print(f'Epoch {epoch+1}/{epochs}: {loss}')

    return loss_vals

    
if __name__ == '__main__':
    file_path = DATA_PATH

    with open(file_path, 'rb') as file:
        data = pkl.load(file)

    config_path = CONFIG_PATH
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_pytorch_network(data,config)


