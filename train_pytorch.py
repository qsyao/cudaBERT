#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
from tqdm import tqdm
import torch
import time
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--input_file",
                    default="./data/deepqa_train_10w.tsv",
                    type=str,
                    help="the input file to predict")
parser.add_argument("--max_seq_length",
                    default=128,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--batch_size",
                    default=8,
                    type=int,
                    help="")
parser.add_argument("--iterator",
                    default=10,
                    type=int,
                    help="")
parser.add_argument("--do_lower_case",
                    default=True,
                    action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--vocab_file",
                    default="./data/vocab.txt",
                    type=str,
                    help="The vocabulary file that the BERT model was trained on.")

args = parser.parse_args()

print("batch_size: {}".format(args.batch_size))
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None):
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
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.length = args.max_seq_length
        self.label = label
        self._get_length()

    def _get_length(self):
        """ Get the useful length of the ids [0 - max_seq_length]"""
        for i in range(len(self.input_mask)):
            index = len(self.input_mask) - 1 - i
            if self.input_mask[index] == 1:
                self.length = index
                break

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

def create_examples(lines):
    examples = []
    for (i, line) in enumerate(lines):
        label = line[1]
        text_a = line[0]
        text_b = ""
        examples.append(
            InputExample(text_a=text_a, text_b=text_b, label=label))
    return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example in tqdm(examples, desc="get features"):
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
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label = example.label))
    return features

with open('/home/wenxh/zyc/bert_train/cuda_bert/data/deepqa_train_10w.tsv', "r", encoding="utf8") as f:
    data = f.readlines()
    lines = []
    for line in data:
        raw = line
        line = line.replace("\0", '').rstrip()
        split_line = line.split('\t')
        label = split_line[-1]
        split_line = split_line[:-1]
        lines.append(['\t'.join(split_line), int(label)])

examples = create_examples(lines)
tokenizer = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=args.do_lower_case)
features = convert_examples_to_features(
    examples, args.max_seq_length, tokenizer)

input_ids = []
segment_ids = []
input_mask = []
classes = []

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model = model.cuda()
min_loss = 1e18
def adjust_learning_rate(optimizer, decay_rate=.99):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

optimizer = optim.SGD(model.parameters(), lr=0.001)

outputRand = 0

for round in range(args.iterator):
    pre_end = -1
    now_loss = 0
    cnt = 0
    for (id, feature) in enumerate(features):
        input_ids.append(feature.input_ids)
        segment_ids.append(feature.segment_ids)
        input_mask.append(feature.input_mask)
        classes.append(feature.label)
        pre_end += 1
        if pre_end % args.batch_size == args.batch_size - 1:
            # print("input_ids: {}".format(input_ids))
            # print("segment_ids: {}".format(segment_ids))
            # print("input_mask: {}".format(input_mask))
            input_ids_tensor = torch.tensor(input_ids)
            segment_ids_tensor = torch.tensor(segment_ids)
            input_mask_tensor = torch.tensor(input_mask)
            classes_tensor = torch.tensor(classes)

            input_ids = []
            segment_ids = []
            input_mask = []
            classes = []

            # TODO:注释
            # model.eval()

            input_ids_tensor = input_ids_tensor.cuda()
            segment_ids_tensor = segment_ids_tensor.cuda()
            input_mask_tensor = input_mask_tensor.cuda()
            classes_tensor = classes_tensor.cuda()

            optimizer.zero_grad()
            class_loss = model(input_ids_tensor, segment_ids_tensor, input_mask_tensor, classes_tensor)
            class_loss.backward()
            optimizer.step()

            now_loss += class_loss * (pre_end + 1)

            pre_end = -1

            if cnt % 200 == 199:
                print("outputRand {}".format(outputRand))
                print("average loss: {} id: {}".format(now_loss / (id+1), id))
                outputRand +=1

            cnt += 1
            # print("class_loss: {}".format(class_loss))

    if len(input_ids) != 0:
        input_ids_tensor = torch.tensor(input_ids)
        segment_ids_tensor = torch.tensor(segment_ids)
        input_mask_tensor = torch.tensor(input_mask)
        classes_tensor = torch.tensor(classes)

        # TODO:注释
        # model.eval()

        input_ids_tensor = input_ids_tensor.cuda()
        segment_ids_tensor = segment_ids_tensor.cuda()
        input_mask_tensor = input_mask_tensor.cuda()
        classes_tensor = classes_tensor.cuda()

        optimizer.zero_grad()
        class_loss = model(input_ids_tensor, segment_ids_tensor, input_mask_tensor, classes_tensor)
        class_loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)

        now_loss += class_loss * (pre_end + 1)

        if cnt % 200 == 199:
            print("outputRand {}".format(outputRand))
            print("average loss: {}".format(now_loss / (id+1)))

    if now_loss < min_loss:
        min_loss = now_loss
    print("Round {}".format(round+1))
    print("***************  now_loss  *************")
    print("***************  {}  *************".format(now_loss / len(features)))
    print("***************  min_loss  *************")
    print("***************  {}  *************".format(min_loss / len(features)))

# optimizer = optim.SGD(model.parameters(), lr=0.000001)
# for i in range(Iters):
#     print("Round: ", i)
#     optimizer.zero_grad()
#     class_loss = model(tokens_tensor, segments_tensors, attention_mask, next_sentence_label)
#     class_loss.backward()
#     optimizer.step()
#     # print("optimizer.lr {}".format(optimizer.lr))
#
#
# nohup python -u train_pytorch.py > tmp1 2>&1 &