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
- Go to ${Project}/cuda_bert
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

- Custom additional layer: In custom.py , take output numpy.array from bert : [batchsize, hidden_size]
- Your own Preprocess functions(define in preprocess.py) to process lines of your own input_file to tagged_line(defined in utils.py), Prepare line_index, line_data(raw string), segment_id, input_id and mask.
- Your funcitons to write line to output_file(defined in utils.py), it takes the raw_line and your output_string as input and returns a string.

### Step 4 
New class engine and set cuda_model, custom_layer, preproecess_function, output_line in main.py

Run main.py

Input your GPU_ID by --gpu 0 1 2 3

### Example
After Step 1 and Step2, we release an example to process ./data/example.tsv to ./data/example.tsv. (Step 3 is set to deal with input file)

The additional layer is Linear + Softmax

```shell
python main.py --input_file ./data/small_v6_label_data.tsv --output_file ./data/test.tsv --gpu 0
```

## Name.txt
Described in name.txt; 

Names of other layers are like layer_0

## Reference
[torch_bert](https://github.com/huggingface/pytorch-pretrained-BERT)

[cnpy](https://github.com/rogersce/cnpy)