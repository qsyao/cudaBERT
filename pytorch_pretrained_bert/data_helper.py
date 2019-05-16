import numpy as np
import logging
import unicodedata
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable


class ReaderDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return vectorize(self.examples[item])

def vectorize(ex):
    input_ids = ex.input_ids
    input_mask = ex.input_mask
    segment_ids = ex.segment_ids

    return input_ids, input_mask, segment_ids, ex.start_positions, ex.end_positions

def batchify(batch):
    batch_input_ids = torch.tensor([ex[0] for ex in batch], dtype=torch.long)
    batch_input_mask = torch.tensor([ex[1] for ex in batch], dtype=torch.long)
    batch_segment_ids = torch.tensor([ex[2] for ex in batch], dtype=torch.long)


    # TODO diff train and dev
    y_s = [ex[3] for ex in batch]
    y_e = [ex[4] for ex in batch]
    yy_s = torch.Tensor(batch_input_ids.size(0), batch_input_ids.size(1)).fill_(0)
    yy_e = torch.Tensor(batch_input_ids.size(0), batch_input_ids.size(1)).fill_(0)
    for i in range(len(batch)):
        for j in range(len(y_s[i])):
            yy_s[i, y_s[i][j]] = 1
            yy_e[i, y_e[i][j]] = 1

    return batch_input_ids, batch_input_mask, batch_segment_ids, Variable(yy_s), Variable(yy_e)



def batchify_eval(batch):
    batch_input_ids = torch.tensor([ex[0] for ex in batch], dtype=torch.long)
    batch_input_mask = torch.tensor([ex[1] for ex in batch], dtype=torch.long)
    batch_segment_ids = torch.tensor([ex[2] for ex in batch], dtype=torch.long)
    batch_example_index = torch.arange(batch_input_ids.size(0), dtype=torch.long)

    return batch_input_ids, batch_input_mask, batch_segment_ids, batch_example_index

