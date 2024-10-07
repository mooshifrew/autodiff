import numpy as np
from autodiff.activation import ReLU, Sigmoid, Linear
from autodiff.network import Network, NetworkParams, PredictionType
import pickle as pkl


file_path = 'autodiff/assignment-one-test-parameters.pkl'

with open(file_path, 'rb') as file:
    data = pkl.load(file)

network_def: NetworkParams = {
    "input_shape": 2,
    "output_shape": 1,
    "prediction_type": PredictionType.REGRESSION,
    "layers": [
        {
            "input_shape": 2,
            "n_neurons": len(data['w1']),
            "weight_init": data['w1'],
            "bias_init": data['b1'] ,
            "activation": ReLU(),
        },
        {
            "input_shape": len(data['w1']),
            "n_neurons": len(data['w2']),
            "weight_init": data['w2'],
            "bias_init": data['b2'] ,
            "activation": ReLU(),
        },
        {
            "input_shape": len(data['w2']),
            "n_neurons": len(data['w3']),
            "weight_init": data['w3'],
            "bias_init": data['b3'] ,
            "activation": Linear(), 
        }
    ]
} 

network = Network(network_def)

def get_loss(y, y_hat): 
    return 0.5*((y_hat - y)**2)

epochs = 3
loss_vals = []

for iteration in range(epochs): 
    print(f'ITERATION {iteration}')
    current_loss = 0
    network.zero_grad()
    for index, input in enumerate(data['inputs']):
        y_pred = network.forward(input)
        print(y_pred)
        y = data['targets'][index] 
        print(f'y: {y}, y_dtype: {y.dtype}, y_pred: {y_pred}')
        loss = get_loss(y, y_pred)
        current_loss += loss

        print(y_pred)
        print(loss)
        print(iteration)

        
        # TODO: I DON'T THINK THIS IS RIGHT SINCE IT SAYS UPDATE PARAMS ONCE PER BATCH BUT THAT MESSES UP LOSS FUNCTION
        network.backward(y_pred - y)
        print("#"*20)
        print("network layers: ", network.layers[0].w)
        print("netowrk grads: ", network.layers[0].w_grads)
        print("#"*20)
    network.update_params(learning_rate=0.01)

        
    loss_vals.append(current_loss/len(data["inputs"]))
   