import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from activation import *
from network import *
from model import SimpleNet, set_model_weights

def loss_fn(y_pred,y_label):
    '''
    
    '''
    loss = 0.5*((y_pred - y_label)**2)

    return loss

def create_network(config: dict, weights:dict):
    input_shape:int = config['NETWORK']['INPUT_SHAPE']
    output_shape:int  = config['NETWORK']['OUTPUT_SHAPE']
    activation: Activation = config['NETWORK']['ACTIVATION']
    is_regression: bool = config['NETWORK']['IS_REGRESSION']

    if is_regression == True:
        classifaction_type: Activation = Linear
    else:
        classifaction_type: Activation = ReLU

    network_def: NetworkParams = {
        "input_shape":input_shape ,
        "output_shape": output_shape,
        "layers": [
            {
                "input_shape": 2,
                "n_neurons": len(weights['w1']),
                "weight_init": weights['w1'],
                "bias_init": weights['b1'] ,
                "activation": activation,
            },
            {
                "input_shape": len(weights['w1']),
                "n_neurons": len(weights['w2']),
                "weight_init": weights['w2'],
                "bias_init": weights['b2'] ,
                "activation": activation,
            },
            {
                "input_shape": len(weights['w2']),
                "n_neurons": len(weights['w3']),
                "weight_init": weights['w3'],
                "bias_init": weights['b3'] ,
                "activation": classifaction_type, 
            }
        ]
    } 

    network = Network(network_def)

    return network
    

def plot_results(loss_vals):
    plt.plot(loss_vals)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show()
    return
    

def train_network(config: dict, network: Network, data: dict):
    epoch = config['TRAIN']['EPOCH']
    lr = config['TRAIN']['LR']
    
    loss_vals = []

    for iteration in range(epoch): 
        current_loss = 0
        network.zero_grad()
        for index, input in tqdm(data['inputs']):
            y_pred = network.forward(input)
            y = data['targets'][index] 
            loss = loss_fn(y, y_pred)
            current_loss += loss
            
            network.backward(y_pred - y)
        print(f"EPOCH: {iteration}, LOSS: {current_loss}")

    network.update_params(learning_rate=lr, data_length=len(data))
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

    def get_loss(y, y_hat):
        return 0.5 * ((y_hat - y) ** 2).mean()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0
        
        for i in range(len(inputs)):
            input_data = inputs[i]
            target_data = targets[i]

            # Forward pass
            output = model(input_data)
            loss = get_loss(output, target_data)
            epoch_loss += loss.item()
            
            # Accumulate gradients
            loss.backward()

        # Perform a single parameter update after accumulating gradients
        optimizer.step()
        
        # Average loss for the epoch
        avg_loss = epoch_loss / len(inputs)
        loss_vals.append(avg_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss}')

        

def compare_results():
    pass
    


