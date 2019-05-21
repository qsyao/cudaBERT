import torch
from torch import nn

import numpy as np
'''
    Model addition to BERT Encoder
    Input:
    numpy.array at CPU : [batchsize, hiddensize]
    Output is defined by user, but must located at CPU

    engine.py will call init_finetune_layer(id_gpu) and
        run(encoding_output) in process_engine_model
    Here is an example
'''

class torch_classify(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(torch_classify, self).__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, pooler_out):
        return self.softmax(self.linear(pooler_out))

class Finetune_Layer(object):
    def __init__(self, is_large):
        self.hidden_size = 1024 if is_large else 768
        self.id_gpu = None
        self.layer = None
    
    def init_finetune_layer(self, id_gpu):
        finetune_layer = torch_classify(2, self.hidden_size)
        custom_kernel = torch.tensor(np.load('model_npy/classifier_kernel.npy').T)
        custom_bias = torch.tensor(np.load('model_npy/classifier_bias.npy'))
        finetune_layer.linear.weight = nn.Parameter(custom_kernel)
        finetune_layer.linear.bias = nn.Parameter(custom_bias)
        self.layer = finetune_layer.cuda('cuda:' + str(id_gpu))
        self.id_gpu = id_gpu
    
    def run(self, encoding_output):
        encoding_output = torch.tensor(encoding_output)\
                                .cuda('cuda:' + str(self.id_gpu))
        with torch.no_grad():
            output = self.layer(encoding_output).cpu().numpy()[:,-1]
        return output