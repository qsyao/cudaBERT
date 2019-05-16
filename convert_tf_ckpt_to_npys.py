import tensorflow as tf

import argparse
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
                    default='tf_checkpoint/' + 'QnA' + '/model.ckpt-333162',
                    type=str,
                    help="Initial checkpoint (usually from a pre-trained BERT model).")

parser.add_argument("--output_dir",
                    default="./model_npy",
                    type=str,
                    help="Output npys pth.")

args = parser.parse_args()




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
    init_vars = tf.train.list_variables(args.init_checkpoint)

    names = []
    arrays = []

    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(args.init_checkpoint, name)
        names.append(name)
        arrays.append(array)

    for i in range(len(names)):
        names[i] = names[i].replace('/', '_')
    for i in range(len(names)):
        numpy.save(args.output_dir  + '/' + names[i], arrays[i])