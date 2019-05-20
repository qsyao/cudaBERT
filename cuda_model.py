import sys
import numpy as np

class cubert(object):
    def __init__(self, id_gpu, args):
        self.id_gpu = id_gpu
        self.max_batchsize = args.batch_size
        self.max_seq_length = args.max_seq_length

        sys.path.insert(0, args.cubert_pth)
        from pybert import load_model, bert

        if args.is_large:
            self.hidden_size = 1024
        else:
            self.hidden_size = 768

        self.model = load_model(args.is_large, args.model_npy_pth, id_gpu,\
                                args.batch_size, args.max_seq_length)
        
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
        
