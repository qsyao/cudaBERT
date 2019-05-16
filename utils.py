import numpy as np

def optimize_batch(tg_line_list):
    max_length = -1
    for tg_line in tg_line_list:
        max_length = max(max_length, tg_line.length)
    batch = []
    for i in range(3):
        new = np.ones([len(tg_line_list), max_length], dtype=np.int32)
        for j in range(len(tg_line_list)):
            new[j] = tg_line_list[j].get_tensor(i)[0:max_length]
        batch.append(new)
    return batch

class Batch_Packed(object):
    '''
        lines_list : list of Tagged_lines
        Tensor : [batchsize, [input_id, seg_id, mask]]
        output : [batchsize, output_custom]
    '''
    def __init__(self, lines_list, tensor):
        self.lines_list = lines_list
        self.tensor = tensor
        self.output = None
    
    def set_output(self, output):
        self.output = output
    
    def write_line(self):
        for i in range(len(self.lines_list)):
            self.lines_list[i].output = self.output[i]

class Tagged_line(object):
    """A single set of features of data."""

    def __init__(self, num_line, line_data, input_ids, input_mask, segment_ids):
        self.num_line = num_line
        self.line_data = line_data
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.length = 512
        self._get_length()
        self.output = None

    def _get_length(self):
        """ Get the useful length of the ids [0 - max_seq_length]"""
        for i in range(len(self.input_mask)):
            index = len(self.input_mask) - 1 - i
            if self.input_mask[index] == 1:
                self.length = index + 1
                break
    
    def get_tensor(self, index):
        if index == 0: 
            return self.input_ids
        if index == 1: 
            return self.segment_ids
        if index == 2: 
            return self.input_mask

def output_line(line_data, output):
    '''
        define by Users to write results to output
        line_data (string): what user use for raw line
        output (string): computation results of bert + custom_layer
    '''
    return line_data + '\t' + output