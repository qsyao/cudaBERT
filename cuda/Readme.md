# Compile

## Before Run Test:

Generate Weights and bias(npy) to model_npy
python convert_pytorch_model_to_npys.npy --bert_config_file /path/to/bert_config --init_checkpoint /path/to/cpkt --output_dir ./model_npy

## Compile

nvcc cuda_bert.cu -o libcubert.so -lcublas -I ./reference/ -lcnpy -L ./ -lz --std=c++11 -shared -Xcompiler -fPIC

## Run test
python pybert.py (from pytorch_pretrained_bert import BertTokenizer Is Needed for tokenlizer)
You can change batchsize and seq_length , Iters or add time record at test()