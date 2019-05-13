#ifndef CUDA_BERT
#define CUDA_BERT

extern "C" {
void bert_train(int batchsize, int seq_length, int nIter, bool base, int num_gpu = 0);
void test_train(int batchsize, int seq_length, int nIter, bool base, int num_gpu = 0);
void test_inference(int batchsize, int seq_length, int nIter, bool base, int num_gpu = 0);
}

#endif
