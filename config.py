import os

class Config(object):
    def __init__(self):
        "Set CUDA Device"
        self.gpu = [0]

        "Using Large Module BERT"
        self.is_large = True

        "The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded."
        self.max_seq_length = 200

        "batch_size."
        self.batch_size = 128

        "Queue Size"
        self.queue_size = 100

        "Too much cache will crash down the memory"
        self.alert_size = 1000000

        "make seq_length_split"
        self.start_split = 50
        self.end_split = 180
        self.split_size = 2

        "skip the first line in input_file."
        self.skip_first_line = False

        "model_npy PATH including addition layer and weights of bert."
        self.model_npy_pth = "./model_npy"

        current_path = os.path.abspath(__file__)
        self.cubert_pth = os.path.abspath(os.path.dirname(current_path)) + "/cuda_bert"

        self.hiddensize = 1024 if self.is_large else 768
    
    