import torch
from torch import nn

import numpy as np
'''
    Model addition to BERT Encoder
    Input:
    torch.tensor at GPU : [batchsize, hiddensize]
    Output is defined by users

    Here is an example
'''

class torch_classify(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(torch_classify, self).__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, pooler_out):
        return self.softmax(self.linear(pooler_out))

def init_custom_layer(hidden_size):
    custom_layer = torch_classify(2, hidden_size)
    custom_kernel = torch.tensor(np.load('model_npy/classifier_kernel.npy').T)
    custom_bias = torch.tensor(np.load('model_npy/classifier_bias.npy'))
    custom_layer.linear.weight = nn.Parameter(custom_kernel)
    custom_layer.linear.bias = nn.Parameter(custom_bias)
    return custom_layer