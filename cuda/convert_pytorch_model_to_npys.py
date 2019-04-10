import argparse
import torch
import numpy   
import os
import errno

parser = argparse.ArgumentParser()

parser.add_argument("--bert_config_file",
                    default="./model_dir/bert_config.json",
                    type=str,
                    help="The config json file corresponding to the pre-trained BERT model. \n"
                         "This specifies the model architecture.")

parser.add_argument("--init_checkpoint",
                    default="./model_dir/pytorch_model_v5.bin",
                    type=str,
                    help="Initial checkpoint (usually from a pre-trained BERT model).")

parser.add_argument("--output_dir",
                    default="./model_dir/pytorch_model_v5.bin",
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

if __name__ == "__main__":
    mkdir_p(args.output_dir)
    dict_model = torch.load(args.init_checkpoint, map_location='cpu')
    for item in dict_model.keys():
        correct_name = convert_name(item, dict_model)
        npy = dict_model[item].numpy()
        if len(npy.shape) == 2 and ('embeddings' not in correct_name):
            new = npy.T.copy()
        else:
            new = npy
        numpy.save(args.output_dir + correct_name, new)
        print(correct_name, " shape : ", new.shape)
    import ipdb; ipdb.set_trace()
