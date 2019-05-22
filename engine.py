import argparse
import copy
import numpy as np
import time
import gc
import sys
from multiprocessing import Process, Queue

from .utils import optimize_batch, Batch_Packed, Tagged_line
from .mylogger import get_mylogger

class batch_container(object):
    """ Init queue with top and base seq_length"""
    def __init__(self, base_length, top_length, batchsize):
        self.base_length = base_length
        self.top_length = top_length
        self.prepare = []
        self.temp = []
        self.min_index = sys.maxsize
        self.batchsize = batchsize
    
    def put(self, tg_line):
        self.temp.append(tg_line)
        if len(self.temp) == self.batchsize:
            self._batch()
    
    def _batch(self):
        """ Batch temp_list and Enqueue """
        # logger.debug("Seg_length in {} - {}: Enqueue batch_size: {}"\
        #               .format(self.base_length, self.top_length, len(self.temp)))
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
    def __init__(self, seq_length_split, list_gpu, logger, batchsize):
        self.num_containers = len(seq_length_split) - 1
        self.boardlines = seq_length_split
        self.dead_queue = Queue(1 + len(list_gpu))
        self.iter = 0
        self.output_queue = Queue()
        self.force_queue = Queue(1)
        self.input_queue = Queue(10)
        self.logger = logger
        self.containers = self._init_containers(batchsize)

    def _init_containers(self, batchsize):
        containers = []
        for i in range(0, self.num_containers):
            containers.append(batch_container(\
                       self.boardlines[i], self.boardlines[i+1], batchsize))
        self.logger.debug("Success Init containers :  *******")
        for container in containers:
            self.logger.debug("Queue.Seq_length : [{}, {}]".format(\
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

class Engine(object):
    def __init__(self, config):
        self.handle = None
        self.finetune_layer = None
        self.cuda_model = None
        self.tokenlizer_line = None
        self.output_line = None
        self.input_file = None
        self.output_file = None
        self.config = config
        self.logger = get_mylogger()
    
    def _init(self):
        self.generate_splits()

        self.handle = queue_manager(self.seq_length_split, self.config.gpu,\
                            self.logger, self.config.batch_size)

        assert(self.cuda_model is not None)
        assert(self.finetune_layer is not None)
        assert(self.tokenlizer_line is not None)
        assert(self.output_line is not None)

        assert(self.config.gpu != [])
                

    # def set_config(self, is_large = True, max_seq_length = 200, batch_size = 128,\
    #                 queue_size = 100, alert_size = 1000000, start_split = 50,\
    #                 end_split = 180, split_size = 2, skip_first_line = False):
    #     self.config.is_large = is_large
    #     self.config.max_seq_length = max_seq_length
    #     self.config.batch_size = batch_size
    #     self.config.queue_size = queue_size
    #     self.config.alert_size = alert_size
    #     self.config.start_split = start_split
    #     self.config.end_split = end_split
    #     self.config.split_size = split_size
    #     self.config.skip_first_line = skip_first_line

    def set_cuda_model(self, cuda_model):
        self.cuda_model = cuda_model

    def set_finetune_layer(self, finetune_layer):
        self.finetune_layer = finetune_layer

    def set_tokenlizer_function(self, tokenlizer_line):
        self.tokenlizer_line = tokenlizer_line

    def set_output_function(self, output_line):
        self.output_line = output_line

    '''
    Splits of Seq_length from [0, max_length]
    Too sparse : Unnecessary Computation from mask
    Too dense : Too much memory cost by cache in post_process
    '''
    def generate_splits(self):
        ret = [0]
        for i in range(self.config.start_split, \
                       self.config.end_split + self.config.split_size, \
                       self.config.split_size):
            ret.append(i)
        ret.append(self.config.max_seq_length)
        self.seq_length_split = ret


    def _engine_model(self, id_gpu):
        config = self.config
        handle = self.handle
        logger = self.logger
        '''
        custom part:
        define and init nn.module
        init weights by numpy
        run after bert_encoding
        '''
        user_layer = self.finetune_layer(config.is_large)
        user_layer.init_finetune_layer(id_gpu)

        model = self.cuda_model(id_gpu, config) 

        start = time.time()
        total_length = 0
        while(1):
            if not handle.input_queue.empty():
                packed_batch = handle.get()
                output = model.encode(packed_batch.tensor)

                output = user_layer.run(output)
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
                                config.input_file, str(total_length), str(end - start)))
        logger.info("engine_model Terminate at gpu" + str(id_gpu))

    def _generate_tagged_line(self, user_output):
        return Tagged_line(
                num_line = user_output[0],
                line_data = user_output[1],
                input_ids = user_output[2],
                input_mask = user_output[3],
                segment_ids = user_output[4])

    def _engine_preprocess(self, input_file):
        config = self.config
        handle = self.handle
        with open(input_file, 'r', encoding='utf-8') as f:
            if self.config.skip_first_line:
                f.readline()
            line = f.readline()
            index = 0
            while line:
                tagged_line = self._generate_tagged_line(self.tokenlizer_line(\
                                                        config.max_seq_length, line, index))
                handle.put(tagged_line)
                index += 1
                line = f.readline()
                if not handle.force_queue.empty():
                    handle.force_queue.get()
                    handle.force_enqueue()
            handle.force_enqueue()
        handle.dead_queue.put("ALL INPUT JOB DONE")
        self.logger.info("engine_preprocess Terminate")


    def _engine_postprocess(self, output_file):
        handle = self.handle
        logger = self.logger
        with open(output_file, 'w', encoding='utf-8') as f:
            start = 0
            end = 0
            window = [""]
            while(1):
                logger.debug("{}, {}".format(handle.output_queue.qsize(), handle.dead_queue.qsize()))
                if handle.output_queue.empty() and handle.dead_queue.qsize() == 1 + len(self.config.gpu):
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
                            window[i] = self.output_line(lines_list[index].line_data, \
                                                                lines_list[index].output)
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
                if end - start > self.config.alert_size and handle.dead_queue.empty():
                    if handle.force_queue.empty():
                        handle.force_queue.put("force_enqueue Now")

                ''' Try to write to File '''
                for line in write_lines:
                    f.write(line + '\n')
                
                del lines_list
                gc.collect()

    def run(self, input_file, output_file):
        assert(type(output_file) == str)
        assert(type(input_file) == str)

        self._init()
        self.config.input_file = input_file
        self.config.output_file = output_file

        file_reader = Process(target=self._engine_preprocess, args=(input_file,))
        file_reader.deamon = True
        file_reader.start()

        runtime_list = []
        for num_gpu in self.config.gpu:
            runtime = Process(target=self._engine_model, args=(num_gpu, ))
            runtime.deamon = True
            runtime.start()
            runtime_list.append(runtime)
        file_writer = Process(target=self._engine_postprocess, args=(output_file,))
        file_writer.deamon = True
        file_writer.start()

        # Terminate
        file_reader.join()
        runtime.join()
        file_writer.join()
        self.logger.warning("ALL JOBS DONE")
