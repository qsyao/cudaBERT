import argparse

import mylogger
from engine import engine
from preprocess import init_tokenlizer, process_line
from custom import custom_layer
from cuda_model import cubert
from utils import output_line

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
logger.info("The Split List of Seq_length")
logger.info(seq_length_split)

if __name__ == "__main__":
    runtime = engine(args, logger, seq_length_split)
    runtime.set_cuda_model(cubert)
    runtime.set_custom_layer(custom_layer)
    runtime.set_preprocess_function(process_line)
    runtime.set_output_function(output_line)

    runtime.run()
