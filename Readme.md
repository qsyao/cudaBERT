# A Fast Muti-processing BERT_Inference System

- BERT Encoder Implentment by CUDA and has been Optimized (Using Kernel Fusion etc)
- Front end is implented by python, for pruning useless sequence length at the end of string.(disabled by mask)
- Tokenlizer and additional layer for BERT_Encoder is implented by Pytorch, Users can define their own additional layer.

4x Faster than Pytorch:

10W lines DataSet on GTX 1080TI

pytorch | CUDA_BERT
---- | ----
2201ms | 506ms

## Constraints
- Nvidia GPUS && nvidia-drivers
- CUDA 9.0
- Cmake > 3.0
- Weights of BERT must be named Correctly (Correct name in name.txt), also correct_name.npy can be generate by checkpoints from [tf_bert](https://github.com/google-research/bert) and [torch_bert](https://github.com/huggingface/pytorch-pretrained-BERT)
## How to Use


### Step 1 
Make libcudaBERT.so
- Go to $(Project)/cuda_bert/cuda_bert
- cmake . && make -j8 

### Step 2

1. Prepare vocab.txt(tokenlizer needed) in ${Project}/model_dir (or input manually)

2. Prepare checkpoints and bert_config_file from tensorflow or pytorch in ${Project}/model_dir  (or input manually)

3. Prepare weights and bias in ${Project}/model_npy
```shell
python convert_pytorch_model_to_npys.py --bert_config_file model_dir/bert_config.json --init_checkpoint model_dir/pytorch_model_v5.bin --output_dir model_npy
```
(or convert_tf_ckpt_to_npys.py )

### Step 3
Define your own functions:

- Custom finetune layer: In apps/finetune.py , take output numpy.array from bert : [batchsize, hidden_size]
```python
class torch_classify(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(torch_classify, self).__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, pooler_out):
        return self.softmax(self.linear(pooler_out))
```
- Your own Tokenlizer functions(define in tokenlizer.py) to process lines of your own input_file to a tuple(Noted in tokenlizer.py), Prepare line_index, line_data(raw string), segment_id, input_id and mask.
```python
def tokenlizer_line(max_seq_length, line, index):
    pass
    return (id_line,
            line_raw_data,
            input_ids,
            input_mask,
            segment_ids)
```
- Your funcitons to write line to output_file(defined in example.py), it takes the raw_line and your output_string as input and returns a string.
```python
def output_line(line_data, output):
    '''
        define by Users to write results to output
        line_data (string): what user use for raw line
        output (string): computation results of bert + custom_layer
    '''
    return line_data + '\t' + str(output)
```

### Step 4 

New class engine , config and set cuda_model, custom_layer, preproecess_function, output_line and config of engine(Noted in config.py) in example.py

The defalt value and meaning of configs are set at config.py.

```python
from cuda_bert.engine import Engine
from cuda_bert.cuda_model import Cuda_BERT

if __name__ == "__main__":
    '''Set Config'''
    config = Engin_Config()
    config.batchsize = 128
    config.model_npy_pth = args.model_npy_pth

    runtime = Engine(config)

    runtime.set_cuda_model(Cuda_BERT)
    runtime.set_finetune_layer(Finetune_Layer)
    runtime.set_tokenlizer_function(tokenlizer_line)
    runtime.set_output_function(output_line)

    runtime.run(args.input_file, args.output_file)
```

Run example.py and Input your GPU_ID by --gpu 0 1 2 3

### Example
After Step 1 and Step2, we release an example to process ./apps/data/example.tsv to ./apps/data/example.tsv. (Step 3 is set to deal with input file)

The additional layer is Linear + Softmax

```shell
cd apps
python example.py --input_file ./data/small_v6_label_data.tsv --output_file ./data/test.tsv --gpu 0
```

## Name.txt
Described in name.txt, and names can't be diffence from names in Name.txt;

Names of other layers are like layer_0

## Retraining

We release a branch for retraining （by cuda)，but it is hard to use for real dataset.  This is more about testing code run time.  Our retraining code run 30% faster than pytorch and tensorflow.

## Reference
[torch_bert](https://github.com/huggingface/pytorch-pretrained-BERT)

[cnpy](https://github.com/rogersce/cnpy)

## Authors

- [Yuchao Zheng](https://github.com/YuchaoZheng)
- [Qingsong Yao](https://github.com/qsyao)

