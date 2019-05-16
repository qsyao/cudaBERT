import argparse
import copy
import numpy as np
import time
import torch
import gc
from multiprocessing import Process, Queue

import mylogger
from custom import init_custom_layer
from utils import optimize_batch, Batch_Packed, output_line
from preprocess import init_tokenlizer, process_line

logger = mylogger.get_mylogger()

parser = argparse.ArgumentParser()

parser.add_argument("--gpu",
                    default=[0],
                    nargs='+',
                    type=int,
                    help="Set CUDA Device")
parser.add_argument("--input_file",
                    default="./data/deepqa_train_10w.tsv",
                    type=str,
                    help="the input file to predict")
parser.add_argument("--output_file",
                    default="./data/engine_10w.tsv",
                    type=str,
                    help="the output file")
parser.add_argument("--skip_first_line",
                    default=False,
                    type=bool,
                    help="skip the first line in input_file.")
parser.add_argument("--max_seq_length",
                    default=200,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--batch_size",
                    default=128,
                    type=int,
                    help="batch_size.")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--vocab_file",
                    default="./model_dir/vocab.txt",
                    type=str,
                    help="The vocabulary file that the BERT model was trained on.")
parser.add_argument("--queue_size",
                    default=100,
                    type=int,
                    help="Queue Size")
parser.add_argument("--alert_size",
                    default=1000000,
                    type=int,
                    help="Too much cache will crash down the memory")
parser.add_argument("--split_size",
                    default=2,
                    type=int,
                    help="split_size")
parser.add_argument('--cubert_pth',
                    type=str,
                    default="../cuda",
                    help="libcubert.so pybert.py PATH.")
parser.add_argument('--model_npy_pth',
                    type=str,
                    default="./model_npy/",
                    help="model_npy PATH including addition layer and weights of bert.")
parser.add_argument('--is_large',
                    type=bool,
                    default=True,
                    help="Using Large Module BERT")
parser.add_argument('--hidden_size',
                    type=int,
                    default=1024,
                    help="hidden_size of BERT Pooler Output")

args = parser.parse_args()

if not args.is_large:
    args.hidden_size = 768

import sys
sys.path.insert(0, args.cubert_pth)
from pybert import load_model, classify, bert

init_tokenlizer(args.vocab_file, args.do_lower_case)

'''
    Splits of Seq_length from [0, max_length]
    Too sparse : Unnecessary Computation from mask
    Too dense : Too much memory cost by cache in post_process
'''
def generate_splits(start, split, end):
    ret = [0]
    for i in range(start, end+split, split):
        ret.append(i)
    ret.append(args.max_seq_length)
    return ret

seq_length_split = generate_splits(50, args.split_size, 180)
logger.debug(seq_length_split)


class batch_container(object):
    """ Init queue with top and base seq_length"""
    def __init__(self, base_length, top_length):
        self.base_length = base_length
        self.top_length = top_length
        self.prepare = []
        self.temp = []
        self.min_index = sys.maxsize
    
    def put(self, tg_line):
        self.temp.append(tg_line)
        if len(self.temp) == args.batch_size:
            self._batch()
    
    def _batch(self):
        """ Batch temp_list and Enqueue """
        logger.debug("Seg_length in {} - {}: Enqueue batch_size: {}"\
                      .format(self.base_length, self.top_length, len(self.temp)))
        if len(self.temp) == 0:
            return 
        batch_tensor = optimize_batch(self.temp)
        packed_batch = Batch_Packed(copy.deepcopy(self.temp), batch_tensor)
        self.prepare.append(packed_batch)
        self.temp.clear()
        self.reset_min_index()

    def force_enqueue(self):
        """ Force to enqueue for two reasons:
        1. limit of the memory
        2. The end of the file
        """
        self._batch()
    
    def pop(self):
        ret = self.prepare.pop()
        self.reset_min_index()
        return ret
    
    def reset_min_index(self):
        if len(self.prepare) != 0:
            self.min_index = self.prepare[0].lines_list[0].num_line
        else:
            self.min_index = sys.maxsize
        return self.min_index

class queue_manager(object):
    """ Handle containers of different seq_length"""
    def __init__(self, seq_length_split):
        self.num_containers = len(seq_length_split) - 1
        self.boardlines = seq_length_split
        self.containers = self._init_containers()
        self.dead_queue = Queue(1 + len(args.gpu))
        self.iter = 0
        self.output_queue = Queue()
        self.force_queue = Queue(1)
        self.input_queue = Queue(10)

    def _init_containers(self):
        containers = []
        for i in range(0, self.num_containers):
            containers.append(batch_container(\
                       self.boardlines[i], self.boardlines[i+1]))
        logger.debug("Success Init containers :  *******")
        for container in containers:
            logger.debug("Queue.Seq_length : [{}, {}]".format(\
                           container.base_length, container.top_length))
        return containers
    
    def put(self, line):
        """ Find the correct queue and enqueue"""
        num_queue = 0
        while line.length > self.boardlines[num_queue+1]:
            num_queue += 1
        self.containers[num_queue].put(line)
        self.enqueue(False)

    def force_enqueue(self):
        for container in self.containers:
            container.force_enqueue()
        while self.enqueue(True) != -1:
            pass

    def enqueue(self, is_force):
        """ 
            Enqueue the min_index batch
            The max_length of the container.prepare list is 4
            return 0 when all prepare list is empty
        """
        max_prepare = 0
        for container in self.containers:
            max_prepare = max(max_prepare, len(container.prepare))
        if max_prepare == 0:
            return -1 # all prepare list clear
        if self.input_queue.empty() or is_force or max_prepare > 4:
            ''' 
                enqueue at 3 situations:
                   1. force enqueue
                   2. At the begining(empty input_queue)
                   3. too many prepared_batch
            '''
            min_index = sys.maxsize
            target_container = None
            for container in self.containers:
                if min_index > container.reset_min_index():
                    min_index = container.reset_min_index()
                    target_container = container
            self.input_queue.put(target_container.pop())
            return min_index 

    def get(self):
        return self.input_queue.get()

    def terminate(self):
        while not self.input_queue.empty():
            time.sleep(1)
        self.input_queue.cancel_join_thread()

def engine_preprocess(handle):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        if args.skip_first_line:
            f.readline()
        line = f.readline()
        index = 0
        while line:
            tagged_line = process_line(args, line, index)
            handle.put(tagged_line)
            index += 1
            line = f.readline()
            if not handle.force_queue.empty():
                handle.force_queue.get()
                handle.force_enqueue()
        handle.force_enqueue()
    handle.dead_queue.put("ALL INPUT JOB DONE")
    logger.info("engine_preprocess Terminate")

def cu_bert(model, input_tensor):
    indexed_tokens = input_tensor[0]
    segments_ids = input_tensor[1]
    attention_mask = input_tensor[2]
    batchsize = indexed_tokens.shape[0]
    seq_length = indexed_tokens.shape[1]
    output = np.ones([batchsize, args.hidden_size]).astype(np.float32)
    bert(model, output, indexed_tokens, segments_ids, \
            batchsize, seq_length, attention_mask)
    return output

# Additional Layer classify implented by CUDA Backending
# num_classes = 2 

# def cu_classify(model, input_tensor, num_classes):
#     indexed_tokens = input_tensor[0]
#     segments_ids = input_tensor[1]
#     attention_mask = input_tensor[2]
#     batchsize = indexed_tokens.shape[0]
#     seq_length = indexed_tokens.shape[1]
#     output = np.ones([batchsize, 2]).astype(np.float32)
#     classify(model, output, indexed_tokens, segments_ids, \
#                 batchsize, seq_length, 2, attention_mask)
#     output = output[:, -1]
#     return output

# def engine_model(handle, num_gpu):
#     model = load_model(True, "./model_npy/", num_gpu) 

#     start = time.time()
#     total_length = 0
#     while(1):
#         if not handle.input_queue.empty():
#             packed_batch = handle.get()
#             output = cu_classify(model, packed_batch.tensor, num_classes)
#             packed_batch.output = output
#             handle.output_queue.put(packed_batch)
#             total_length += packed_batch.tensor[0].shape[0]
#             print("\rProcess Batch : {}".format(total_length), end="", flush=True)
#         elif not handle.dead_queue.empty():
#             handle.dead_queue.put("ALL MODEL JOB DONE")
#             handle.terminate()
#             break
#     end = time.time()
#     logger.warning("Predict File {} total_length: {} Cost: {}".format(  \
#                             args.input_file, str(total_length), str(end - start)))
#     logger.info("engine_model Terminate" + str(num_gpu))

def engine_model(handle, num_gpu):
    '''
    custom part:
    define and init nn.module
    init weights by numpy
    run after bert_encoding
    '''
    custom_layer = init_custom_layer(args.hidden_size)
    custom_layer = custom_layer.cuda('cuda:' + str(num_gpu))
    model = load_model(args.is_large, args.model_npy_pth, num_gpu,\
                                args.batch_size, args.max_seq_length) 

    start = time.time()
    total_length = 0
    while(1):
        if not handle.input_queue.empty():
            packed_batch = handle.get()
            output = cu_bert(model, packed_batch.tensor)
            output = torch.tensor(output).cuda('cuda:' + str(num_gpu))

            with torch.no_grad():
                output = custom_layer(output).cpu().numpy()[:,-1]
            # print(output)
            packed_batch.output = output
            handle.output_queue.put(packed_batch)
            total_length += packed_batch.tensor[0].shape[0]
            print("\rProcess Batch : {}".format(total_length), end="", flush=True)
        elif not handle.dead_queue.empty():
            handle.dead_queue.put("ALL MODEL JOB DONE")
            handle.terminate()
            break
    end = time.time()
    logger.warning("Predict File {} total_length: {} Cost: {}".format(  \
                            args.input_file, str(total_length), str(end - start)))
    logger.info("engine_model Terminate" + str(num_gpu))


def engine_postprocess(handle):
    with open(args.output_file, 'w', encoding='utf-8') as f:
        start = 0
        end = 0
        window = [""]
        while(1):
            logger.debug("{}, {}".format(handle.output_queue.qsize(), handle.dead_queue.qsize()))
            if handle.output_queue.empty() and handle.dead_queue.qsize() == 1 + len(args.gpu):
                logger.info("engine_postprocess Terminate")
                handle.output_queue.cancel_join_thread()
                handle.force_queue.cancel_join_thread()
                break
            
            if handle.output_queue.empty():
                time.sleep(1)
                continue

            packed_batch = handle.output_queue.get()
            packed_batch.write_line()
            lines_list = packed_batch.lines_list

            new_end = lines_list[-1].num_line + 1
            index = 0
            logger.debug("Start {} End {} New_end {}".format(start, end, new_end))
            for i in range(new_end - start):
                if i + start >= end:
                    window.append("")
                    end += 1
                if window[i] != "":
                    if lines_list[index].num_line == start + i:
                        logger.error("Line Index {} is written twice!".format(start + i))
                        return 
                    else:
                        logger.debug("windew[{}] has been set".format(start + i))
                        continue
                else:
                    if lines_list[index].num_line == start + i:
                        window[i] = output_line(lines_list[index].line_data, \
                                                    str(lines_list[index].output))
                        index += 1
                    else:
                        logger.debug("window[{}] Not in this batch".format(start + i))
                        continue 
                logger.debug("Window[{}], index:{}".format(start + i, index))        
            ''' Find the length of the lines prepared '''
            write_length = 0
            for i in range(end - start):
                if window[i] != "":
                    write_length = i + 1
                else:
                    break

            logger.debug("Writing line {} - {} to File".format(start, start+write_length))
            # if write_length > 0:
            #     print("\r                                 Write Lines : {}".\
            #                         format(start+write_length), end="", flush=True)
            write_lines = window[:write_length]
            window = copy.deepcopy(window[write_length:])
            start += write_length
            
            ''' force enqueue '''
            if end - start > args.alert_size and handle.dead_queue.empty():
                if handle.force_queue.empty():
                    handle.force_queue.put("force_enqueue Now")

            ''' Try to write to File '''
            for line in write_lines:
                f.write(line + '\n')
            
            del lines_list
            gc.collect()

if __name__ == "__main__":
    handle = queue_manager(seq_length_split)

    file_reader = Process(target=engine_preprocess, args=(handle, ))
    file_reader.deamon = True
    file_reader.start()

    runtime_list = []
    for num_gpu in args.gpu:
        runtime = Process(target=engine_model, args=(handle, num_gpu, ))
        runtime.deamon = True
        runtime.start()
        runtime_list.append(runtime)
    file_writer = Process(target=engine_postprocess, args=(handle, ))
    file_writer.deamon = True
    file_writer.start()

    # Terminate
    file_reader.join()
    runtime.join()
    file_writer.join()
    logger.warning("ALL JOBS DONE")

