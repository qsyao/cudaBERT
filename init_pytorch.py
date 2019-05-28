
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification

import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--bert_config_file",
                    default="./model_dir/bert_config.json",
                    type=str,
                    help="The config json file corresponding to the pre-trained BERT model. \n"
                         "This specifies the model architecture.")
parser.add_argument("--is_large",
                    default=0,
                    type=int,
                    help="is_large")
parser.add_argument("--init_checkpoint",
                    default="./model_dir/pytorch_model_v5.bin",
                    type=str,
                    help="Initial checkpoint (usually from a pre-trained BERT model).")

parser.add_argument("--input_file",
                    default="./data/preprocess_deepqa_train_10w.tsv",
                    type=str,
                    help="the input file to predict")
parser.add_argument("--skip_first_line",
                    default=False,
                    type=bool,
                    help="skip the first line.")

parser.add_argument("--question_index",
                    default=0,
                    type=int,
                    help="question_index in input_file")
parser.add_argument("--document_index",
                    default=1,
                    type=int,
                    help="document_index in input_file")

parser.add_argument("--predict_output_file",
                    default="./data/predict_output.tsv",
                    type=str,
                    help="predict data output")

parser.add_argument("--task_name",
                    default="mrpc",
                    type=str,
                    help="The name of the task to train.")
parser.add_argument("--vocab_file",
                    default="vocab.txt",
                    type=str,
                    help="The vocabulary file that the BERT model was trained on.")
parser.add_argument("--train_file",
                    default="deepqa_train_10w.tsv",
                    type=str,
                    help="The vocabulary file that the BERT model was trained on.")

parser.add_argument("--output_dir",
                    default="/data/",
                    type=str,
                    help="The output directory where the model checkpoints will be written.")

## Other parameters
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--max_seq_length",
                    default=200,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=True,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--use_gpu",
                    default=False,
                    action='store_true',
                    help="Whether to run eval on the dev set.")

parser.add_argument("--train_batch_size",
                    default=8,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument("--save_checkpoints_steps",
                    default=100000,
                    type=int,
                    help="How often to save the model checkpoint.")
parser.add_argument("--dev_steps",
                    default=50000,
                    type=int,
                    help="How often to save the model checkpoint.")
parser.add_argument("--log_steps",
                    default=2000,
                    type=int,
                    help="How often to save the model checkpoint.")

parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
parser.add_argument("--accumulate_gradients",
                    type=int,
                    default=1,
                    help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumualte before performing a backward/update pass.")
parser.add_argument('--iters',
                    type=int,
                    default=1,
                    help="Use cubert to compute faster.")
args = parser.parse_args()

str_large = "large" if args.is_large else "base"

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, line_data, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.line_data = line_data
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, line_data, input_ids, input_mask, segment_ids, label):
        self.line_data = line_data
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            # modify by yz
            # reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            data = f.readlines()
            lines = []
            for line in data:
                line = line.replace("\0", '').rstrip()
                split_line = line.split('\t')
                lines.append(split_line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""


    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, args.dev_file)), "dev")

    def get_predict_examples(self, input_file):
        """See base class."""
        return self._create_examples(
            self._read_tsv(input_file), "predict")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and args.skip_first_line:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(line_data='\t'.join(line), guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples




def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for example in tqdm(examples, desc="prepare_data"):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
            InputFeatures(
                line_data=None,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label = int(example.label)))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    # print(outputs==labels)
    # print(type(outputs==labels))
    return np.sum(outputs == labels)


def save_model(model, save_dir, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = "steps_{}.pt".format(steps)
    save_path = os.path.join(save_dir, save_prefix)
    torch.save(model.state_dict(), save_path)
    """output_dir save_checkpoints_steps dev_steps"""

def batch_numpy(indexed_tokens, segments_ids, attention_mask):
    total_size = indexed_tokens.shape[0]
    assert(indexed_tokens.shape[0] == segments_ids.shape[0])
    assert(segments_ids.shape[0] == attention_mask.shape[0])
    batchsize = args.eval_batch_size
    batch = []
    pointer = 0
    while pointer < total_size:
        offset = min(batchsize, total_size - pointer)
        target = [indexed_tokens[pointer:pointer+offset], 
                  segments_ids[pointer:pointer+offset],
                  attention_mask[pointer:pointer+offset]]
        batch.append(target)
        pointer += offset
    return batch

def get_max_length(mask):
    for i in range(mask.shape[0]):
        index = mask.shape[0] - 1 - i
        if mask[index] == 1:
            return index
    return 0

def optimize_batch(eval_data):
    for k in tqdm(range(len(eval_data)), desc="optimize_batch"):
        mask = eval_data[k][2]
        max_length = -1
        for i in range(mask.shape[0]):
            max_length = max(max_length, get_max_length(mask[i]))
        for i in range(3):
            new = np.ones([mask.shape[0], max_length+1], dtype=np.int32)
            for j in range(mask.shape[0]):
                new[j] = eval_data[k][i][j][0:max_length+1]
            eval_data[k][i] = new
    return eval_data
         
def adjust_learning_rate(optimizer, decay_rate=.99):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def main():
    processors = {
        "mrpc": MrpcProcessor,
    }


    # logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)

    model = BertForSequenceClassification.from_pretrained_init( \
                    'bert-' + str_large + '-uncased', num_labels=2)
    model.train()
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    eval_examples = processor.get_predict_examples(args.train_file)

    # if os.path.isfile(eval_cache_path + "asd"):
    #     logging.info('Found cache: %s, loading' % eval_cache_path)
    #     eval_features = load_from_pkl(eval_cache_path)
    # else:
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer)
        # dump_to_pkl(eval_features, eval_cache_path)

    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(eval_examples))
    # logger.info("  Batch size = %d", args.eval_batch_size)
    #
    # all_input_lines = [f.line_data for f in eval_examples]
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size)


    import time

    temp_time = 0
    temp_loss = 0
    Iter = 0
    while(1):
        for inputs_ids, input_masks, segments_ids, labels in eval_dataloader:

            inputs_ids = inputs_ids.cuda()
            input_masks = input_masks.cuda()
            segments_ids = segments_ids.cuda()
            labels = labels.cuda()

            start = time.time()

            optimizer.zero_grad()
            iter_loss = model(inputs_ids, input_masks, segments_ids, labels)
            iter_loss.backward()
            optimizer.step()
            end = time.time()

            temp_time += end - start
            temp_loss += iter_loss.data

            if(Iter < 10):
                print(iter_loss.data)
            
            if Iter % 50 == 0:
                print("Iter: {} , Avg_Loss: {} , Time : {}ms".format(Iter, \
                                                    temp_loss/50, temp_time * 1000 /50))
                temp_loss = 0
                temp_time = 0
            Iter += 1
            
 

if __name__ == "__main__":
    main()
