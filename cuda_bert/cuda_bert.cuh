#ifndef CUDA_BERT
#define CUDA_BERT

extern "C"
void test(int batchsize, int seq_length, int nIter, bool base, int num_gpu=0);

#endif
