import argparse

import sys
sys.path.append("../../")

from cuda_bert.engine import Engine
from tokenlizer import init_tokenlizer, tokenlizer_line
from finetune import Finetune_Layer
from cuda_bert.cuda_model import Cuda_BERT

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

init_tokenlizer(args.vocab_file, args.do_lower_case)

def output_line(line_data, output):
    '''
        define by Users to write results to output
        line_data (string): what user use for raw line
        output (string): computation results of bert + custom_layer
    '''
    return line_data + '\t' + str(output)

if __name__ == "__main__":
    runtime = Engine()

    '''Set Config'''
    runtime.set_config()
    runtime.config.model_npy_pth = args.model_npy_pth

    runtime.set_cuda_model(Cuda_BERT)
    runtime.set_finetune_layer(Finetune_Layer)
    runtime.set_tokenlizer_function(tokenlizer_line)
    runtime.set_output_function(output_line)

    runtime.run(args.input_file, args.output_file)
