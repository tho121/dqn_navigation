import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN_model(nn.Module):
    """description of class"""

    def __init__(self, input_size, output_size, nodes_per_layer_count):


        super(DQN_model, self).__init__()

        if(len(nodes_per_layer_count) == 0):
            return

        tempLayers = list()
        tempLayers.append(nn.Linear(input_size, nodes_per_layer_count[0]))

        lastIndex = len(nodes_per_layer_count)
        for i in range(1, lastIndex):
            tempLayers.append(nn.Linear(nodes_per_layer_count[i - 1], nodes_per_layer_count[i]))

        
        tempLayers.append(nn.Linear(nodes_per_layer_count[lastIndex - 1], output_size))

        self.layers = nn.ModuleList(tempLayers)

    def forward(self, state):
        
        x = F.relu(self.layers[0](state))

        lastIndex = len(self.layers) - 1
        for i in range(1, lastIndex):
            x = F.relu(self.layers[i](x))

        return self.layers[lastIndex](x)

    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
                
