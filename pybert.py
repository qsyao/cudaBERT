from ctypes import *
from pytorch_pretrained_bert import BertTokenizer
import numpy
from numpy.ctypeslib import ndpointer 

class Retval(Structure):
    _fields_ = [("tensor", ndpointer(numpy.float)),
                ("Pooled_Output", ndpointer(numpy.float))]

import os
lib_dir = os.path.abspath(os.path.dirname(__file__))
lib = CDLL(lib_dir + "/libcudaBERT.so", RTLD_GLOBAL)

init_model = lib.init_model
init_model.argtypes = [c_bool, c_int, c_char_p]
init_model.restype = c_void_p

def load_model(is_large_model, model_dir, num_gpu=0):    
    return init_model(is_large_model, num_gpu, bytes(model_dir, encoding='utf-8'))

inference = lib.Cuda_Inference
inference.argtypes = [c_void_p, ndpointer(numpy.int32), ndpointer(numpy.int32),\
                      c_int, c_int, ndpointer(numpy.int32)]
inference.restype = Retval

classify = lib.Cuda_Classify
classify.argtypes = [c_void_p, ndpointer(numpy.float32), ndpointer(numpy.int32), ndpointer(numpy.int32),\
                      c_int, c_int, c_int, ndpointer(numpy.int32)]

def filling_inputs(words, seq_length):
    length = ( len(words) - 1 )
    total = length - 1
    end_word  = words[-1]
    words = words[:-1]
    while total < seq_length - 2:
        total = total + 1
        words.append(words[total%length])
    words.append(end_word)
    return words

def batch_inputs(input, batchsize):
    ret = []
    for i in range(batchsize):
        ret.append(input)
    return ret

def warp_inputs(indexed_tokens, segments_ids, attention_masks):
    c_index = numpy.array(indexed_tokens).astype(numpy.int32)
    c_seg = numpy.array(segments_ids).astype(numpy.int32)
    c_mask = numpy.array(attention_masks).astype(numpy.int32)
    return c_index, c_seg, c_mask


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def test(is_large, model_dir, batchsize, seq_length, num_gpu=0):
    Iters = 100
    #assert( seq_length * batchsize < 80 * 1000, "Seq_length * Batchsize is too large")
    max_length = batchsize * seq_length

    tokenized_text = ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']
    segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    attention_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

    tokenized_text = filling_inputs(tokenized_text, seq_length)
    segments_ids = filling_inputs(segments_ids, seq_length)
    attention_mask = filling_inputs(attention_mask, seq_length)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    indexed_tokens = batch_inputs(indexed_tokens, batchsize)
    segments_ids = batch_inputs(segments_ids, batchsize)
    attention_mask = batch_inputs(attention_mask, batchsize)

    c_segments_ids = numpy.array(segments_ids).astype(numpy.int32)
    c_indexed_tokens = numpy.array(indexed_tokens).astype(numpy.int32)
    c_attention_mask = numpy.array(attention_mask).astype(numpy.int32)

    model = load_model(is_large, model_dir, num_gpu=num_gpu)

    import time
    start = time.time()
    for i in range(Iters):
        ret = inference(model, c_indexed_tokens, c_segments_ids, batchsize, seq_length, c_attention_mask)
    end = time.time()
    print("Batchsize: {} Seq_length: {} Use_Time: {}".format(batchsize, seq_length, (end - start)/Iters))

if __name__ == "__main__":
    test(False, "model_npy/base_uncased", 1, 128)





# check_model = lib.check_model
# check_model.argtypes = [c_void_p]
# check_model(model)
# check_inputs = lib.check_inputs
# check_inputs.argtypes = [ndpointer(numpy.int32), c_int]
# check_inputs(c_indexed_tokens, max_length)
# check_inputs(c_segments_ids, max_length)
# check_inputs(c_attention_mask, max_length)




# import ipdb; ipdb.set_trace()
