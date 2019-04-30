# Compile

## Before Run Test:

Generate Weights and bias(npy) to model_npy
python convert_pytorch_model_to_npys.py --bert_config_file /path/to/bert_config --init_checkpoint /path/to/cpkt --output_dir ./model_npy

E.g.,
Download bert-base-uncased PyTorch pre-trained model: 
```
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar xvf bert-base-uncased.tar.gz 
python3 convert_pytorch_model_to_npys.py --bert_config_file ./bert_config.json --init_checkpoint ./pytorch_model.bin --output_dir ./model_npy/base_uncased/
```

## Compile cudaBERT library
```
cmake .
make
```
## Run test

### Test through PyTorch:
```
python pybert.py 
```
(from pytorch_pretrained_bert import BertTokenizer Is Needed for tokenlizer)
You can change batchsize and seq_length , Iters or add time record at test()

### Test with c++ api:
```
g++ ./test/test.cpp -o bert_test -lcudaBERT -L ./ --std=c++11
./bert_test
```