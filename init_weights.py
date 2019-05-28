import argparse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
import os
import errno
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--is_large",
                    default=1,
                    type=int,
                    help="is_large")
parser.add_argument("--output_dir",
                    default="",
                    type=str,
                    help="Output npys.")
args = parser.parse_args()

def convert_name(name, dict_model):
    if 'LayerNorm' in name:
        if 'weight' in name:
            name = name.replace('weight', 'gamma')
        else:
            name = name.replace('bias', 'beta')
    else:
        if len(dict_model[name].shape) == 2:
            name = name.replace('weight', 'kernel')
    if 'embeddings' in name:
        if '_kernel' in name:
            name = name.replace('_kernel', '')
    name = name.replace('.', '_')
    return name

def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e

str_large = "large" if args.is_large else "base"
temp = "temp_init_"+str_large+".bin"
if args.output_dir == "":
    args.output_dir = "model_npy/init_" + str_large
print("init model_npy for bert {} model in {}".format(str_large, args.output_dir))


if __name__ == "__main__":
    model = BertForSequenceClassification.from_pretrained_init( \
                    'bert-' + str_large + '-uncased', num_labels=2)
    torch.save(model.state_dict(), temp)
    mkdir_p(args.output_dir)
    dict_model = torch.load(temp, map_location='cpu')
    for item in dict_model.keys():
        correct_name = convert_name(item, dict_model)
        npy = dict_model[item].numpy()
        if len(npy.shape) == 2 and ('embeddings' not in correct_name):
            new = npy.T.copy()
        else:
            new = npy
        np.save(args.output_dir + "/" + correct_name, new)
        print(correct_name, " shape : ", new.shape)
