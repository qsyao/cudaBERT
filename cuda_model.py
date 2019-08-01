import sys
import numpy as np

class Cuda_BERT(object):
    def __init__(self, id_gpu, config):
        self.id_gpu = id_gpu
        self.max_batchsize = config.batch_size
        self.max_seq_length = config.max_seq_length

        sys.path.insert(0, config.cubert_pth)
        from pybert import load_model, bert

        if config.is_large:
            self.hidden_size = 1024
        else:
            self.hidden_size = 768

        self.model = load_model(config.is_large, config.model_npy_pth, id_gpu,\
                                config.batch_size, config.max_seq_length)
        
        self.cu_encode = bert

    def encode(self, input_tensor):
        '''
        Input_tensor:
            inputs_id, segments_id, mask:
            numpy.array [batchsize, seq_length]
        '''
        indexed_tokens = input_tensor[0]
        segments_ids = input_tensor[1]
        attention_mask = input_tensor[2]
        batchsize = indexed_tokens.shape[0]
        seq_length = indexed_tokens.shape[1]
        output = np.ones([batchsize, self.hidden_size]).astype(np.float32)
        self.cu_encode(self.model, output, indexed_tokens, segments_ids, \
                            batchsize, seq_length, attention_mask)
        return output
        
    def __del__(self):
        from pybert import unload_model
        unload_model(self.model)
