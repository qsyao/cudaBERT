#include "cuda_runtime.h"
#include <time.h>
#include <iostream>
#include <iomanip>

#include "utils/common.h"
#include "utils/bert.cuh"
#include "utils/tokenization.h"
#include "cuda_bert.cuh"

int *filling_inputs(int *tensor, int seq_length, int start_length, int batchsize) {
    int *target = (int *) malloc(sizeof(int) * seq_length * batchsize);
    for (int i = 0; i < seq_length - 1; i++) {
        target[i] = tensor[i % (start_length - 1)];
    }
    target[seq_length - 1] = tensor[start_length - 1];
    for (int i = 1; i < batchsize; i++) {
        memcpy(target + seq_length * i, target, seq_length * sizeof(int));
    }
    return target;
}

int *anthor_filling_inputs(int *tensor, int seq_length, int start_length, int batchsize) {
    int *target = (int *) malloc(sizeof(int) * seq_length * batchsize);
    for (int i = 0; i < start_length; i++) {
        target[i] = tensor[i];
    }
    for (int i = start_length; i < seq_length; i++) {
        target[i] = 0;
    }
    for (int i = 1; i < batchsize; i++) {
        memcpy(target + seq_length * i, target, seq_length * sizeof(int));
    }
    return target;
}

void *cubert_open_tokenizer(const char *vocab_file, bool do_lower_case) {
    return new FullTokenizer(vocab_file, do_lower_case);
}

void cubert_close_tokenizer(void *tokenizer) {
    delete (FullTokenizer *) tokenizer;
}

/**
 * Truncates a sequence pair in place to the maximum length.
 * @param tokens_a
 * @param tokens_a
 * @param max_length
 */
void _truncate_seq_pair(std::vector <std::string> *tokens_a,
                        std::vector <std::string> *tokens_b,
                        size_t max_length) {
// This is a simple heuristic which will always truncate the longer sequence
// one token at a time. This makes more sense than truncating an equal percent
// of tokens from each, since if one sequence is very short then each token
// that's truncated likely contains more information than a longer sequence.
    while (true) {
        size_t total_length = tokens_a->size() + tokens_b->size();
        if (total_length <= max_length) {
            break;
        }
        if (tokens_a->size() > tokens_b->size()) {
            tokens_a->pop_back();
        } else {
            tokens_b->pop_back();
        }
    }
}

/**
 * Converts a single `InputExample` into a single `InputFeatures`.
 */
void convert_single_example(FullTokenizer *tokenizer,
                            size_t max_seq_length,
                            const char *text_a, const char *text_b,
                            int *input_ids, int *input_mask, int *segment_ids) {
    std::vector <std::string> tokens_a;
    tokens_a.reserve(max_seq_length);

    std::vector <std::string> tokens_b;
    tokens_b.reserve(max_seq_length);

    tokenizer->tokenize(text_a, &tokens_a, max_seq_length);
    if (text_b != nullptr) {
        tokenizer->tokenize(text_b, &tokens_b, max_seq_length);

        // Modifies `tokens_a` and `tokens_b` in place so that the total
        // length is less than the specified length.
        // Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(&tokens_a, &tokens_b, max_seq_length - 3);
    } else {
        if (tokens_a.size() > max_seq_length - 2) {
            tokens_a.resize(max_seq_length - 2);
        }
    }

    // The convention in BERT is:
    // (a) For sequence pairs:
    //  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    //  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    // (b) For single sequences:
    //  tokens:   [CLS] the dog is hairy . [SEP]
    //  type_ids: 0     0   0   0  0     0 0
    //
    // Where "type_ids" are used to indicate whether this is the first
    // sequence or the second sequence. The embedding vectors for `type=0` and
    // `type=1` were learned during pre-training and are added to the wordpiece
    // embedding vector (and position vector). This is not *strictly* necessary
    // since the [SEP] token unambiguously separates the sequences, but it makes
    // it easier for the model to learn the concept of sequences.
    //
    // For classification tasks, the first vector (corresponding to [CLS]) is
    // used as as the "sentence vector". Note that this only makes sense because
    // the entire model is fine-tuned.
    input_ids[0] = tokenizer->convert_token_to_id("[CLS]");
    segment_ids[0] = 0;
    for (int i = 0; i < tokens_a.size(); ++i) {
        input_ids[i + 1] = tokenizer->convert_token_to_id(tokens_a[i]);
        segment_ids[i + 1] = 0;
    }
    input_ids[tokens_a.size() + 1] = tokenizer->convert_token_to_id("[SEP]");
    segment_ids[tokens_a.size() + 1] = 0;

    if (text_b != nullptr) {
        for (int i = 0; i < tokens_b.size(); ++i) {
            input_ids[i + tokens_a.size() + 2] = tokenizer->convert_token_to_id(tokens_b[i]);
            segment_ids[i + tokens_a.size() + 2] = 1;
        }
        input_ids[tokens_b.size() + tokens_a.size() + 2] = tokenizer->convert_token_to_id("[SEP]");
        segment_ids[tokens_b.size() + tokens_a.size() + 2] = 1;
    }

    size_t len = text_b != nullptr ? tokens_a.size() + tokens_b.size() + 3 : tokens_a.size() + 2;
    std::fill_n(input_mask, len, 1);

    // Zero-pad up to the sequence length.
    std::fill_n(input_ids + len, max_seq_length - len, 0);
    std::fill_n(input_mask + len, max_seq_length - len, 0);
    std::fill_n(segment_ids + len, max_seq_length - len, 0);
}

void convert_batch_example(void *tokenizer, int batch_size,
                           int max_seq_length,
                           std::vector<std::string> &text_a,
                           std::vector<std::string> &text_b,
                           std::vector<int> &gt_classes,
                           std::vector<int> &text_id,
                           int *input_ids, int *segment_ids, int *input_mask, int *classes) {
    //TODO:Cut string len

    for (int batch_idx = 0; batch_idx < text_id.size(); ++batch_idx) {
        classes[batch_idx] = gt_classes[text_id[batch_idx]];
        convert_single_example((FullTokenizer *) tokenizer,
                               max_seq_length,
                               text_a[text_id[batch_idx]].c_str(),
                               text_b.size() == 0 ? nullptr : text_b[text_id[batch_idx]].c_str(),
                               input_ids + max_seq_length * batch_idx,
                               input_mask + max_seq_length * batch_idx,
                               segment_ids + max_seq_length * batch_idx);
    }
}


extern "C" {

bert *init_model(bool large = false, int num_gpu = 0, std::string dir = "", bool is_train = false, float lr = 0.001,
                 std::string optim = "sgd", bool optimRunningTime = true, int num_classes = 2) {
    bert *ret = new bert(large, num_gpu, dir, is_train, optimRunningTime, num_classes, optim, lr);
    return ret;
}

Retval Cuda_Inference(bert *model,
                      int *words,
                      int *token_types,
                      int batchsize,
                      int seq_length,
                      int *masks) {
    model->BERT_Inference(words,
                          token_types,
                          batchsize,
                          seq_length,
                          masks);

    return model->ret;
}

void Cuda_Classify(bert *model,
                   float *output,
                   int *words,
                   int *token_types,
                   int batchsize,
                   int seq_length,
                   int num_classes,
                   int *attention_mask) {
    model->BERT_Inference(words,
                          token_types,
                          batchsize,
                          seq_length,
                          attention_mask);
    float *output_gpu;
    output_gpu = model->classify_inference(model->ret.pooled_output, num_classes);
    model->get_gpu_result(output, output_gpu, batchsize * num_classes);
}

float cuda_classify_train(bert *model,
                         int *words,
                         int *token_types,
                         int *classes,
                         int batchsize,
                         int seq_length,
                         int num_classes,
                         int *attention_mask) {
    model->BERT_train(words,
                      token_types,
                      batchsize,
                      seq_length,
                      attention_mask);
    float output_gpu = model->classify_train(classes, model->ret.pooled_output, num_classes);
    return output_gpu;
}

void test_inference(int batchsize, int seq_length, int nIter, bool base, int num_gpu) {
    bert *model = init_model(base, num_gpu);

    int test_word_id_seed[11] = {2040, 2001, 3958, 27227, 1029, 3958, 103,
                                 2001, 1037, 13997, 11510};
    int test_token_type_id_seed[11] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

    int attention_mask[11] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    int classes[4] = {1, 1, 1, 1};

    int *test_word_id, *test_token_type_id, *test_attention_mask;
    test_word_id = anthor_filling_inputs(test_word_id_seed, seq_length, 11, batchsize);
    test_token_type_id = anthor_filling_inputs(test_token_type_id_seed, seq_length, 11, batchsize);
    test_attention_mask = anthor_filling_inputs(attention_mask, seq_length, 11, batchsize);
    std::cout << " Seq_length : " << seq_length << std::endl;
    std::cout << " Batchsize : " << batchsize << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *output_pinned;
    checkCudaErrors(cudaMallocHost((void **) &output_pinned,
                                   (1024) * model->handle->hidden_size * sizeof(float)));

    //Warm Up
    for (int i = 0; i < 10; i++) {
        model->BERT_Inference(
                test_word_id,
                test_token_type_id,
                batchsize,
                seq_length,
                test_attention_mask);
        model->get_gpu_result(output_pinned,
                              model->ret.pooled_output,
                              model->handle->batchsize * model->handle->hidden_size);

        if (i == 0) {
            debug_tensor<float>(std::string("unit_test"),
                                output_pinned,
                                10,
                                model->handle->hidden_size,
                                max(model->handle->batchsize / 10, (long) 1));
        }
    }

    double total_time = 0;
    for (int i = 0; i < nIter; i++) {
        float it_time;
        cudaEventRecord(start);
        float *output;
        // cuda_classify(
        //         model,
        //         output,
        //         test_word_id,
        //         test_token_type_id,
        //         classes,
        //         batchsize,
        //         seq_length,
        //         2,
        //         test_attention_mask
        // );
        model->BERT_Inference(
                test_word_id,
                test_token_type_id,
                batchsize,
                seq_length,
                test_attention_mask);

        model->get_gpu_result(output_pinned,
                              model->ret.pooled_output,
                              model->handle->batchsize * model->handle->hidden_size);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&it_time, start, stop);
        total_time += it_time;
    }

    delete model;

    double dSeconds = total_time / (double) nIter;
    printf("Time= %.2f(ms)\n", dSeconds);
}

int generateSeed(int i) {
    return rand() % i;
}

void bert_train(int batchsize, int seq_length, int nIter, bool base, int num_gpu) {

    bert *model = init_model(base, num_gpu, "", true);

    int input_ids[batchsize * seq_length];
    int input_mask[batchsize * seq_length];
    int segment_ids[batchsize * seq_length];
    int classes[batchsize];

    std::cout << " Seq_length : " << seq_length << std::endl;
    std::cout << " Batchsize : " << batchsize << std::endl;

    std::vector <std::string> text_a;
    std::vector <std::string> text_b;
    std::vector<int> gt_classes;
    read_tsv("/home/wenxh/zyc/bert_train/cuda_bert/data/deepqa_train_10w.tsv", text_a, gt_classes);

    int tot_line_len = text_a.size();
    srand(time(NULL));
    std::vector<int> text_id(tot_line_len);
    for (int i = 0; i < tot_line_len; i++)
        text_id[i] = i;
    void *tokenizer = cubert_open_tokenizer("/home/wenxh/zyc/bert_train/cuda_bert/data/vocab.txt", true);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double total_time = 0;
    double min_loss = 1e18;
    int num_labels = 2;
    int outputRand = 0;
    for (int i = 0; i < nIter; i++) {
//         TODO: random
//        random_shuffle(text_id.begin(), text_id.end(), generateSeed);
        double now_loss = 0;
        for (int j = 0; j < tot_line_len / batchsize + (tot_line_len % batchsize == 0 ? 0 : 1); j++) {
            std::vector<int> tmp_text_id;
            for (int k = j * batchsize; k < min((j + 1) * batchsize, tot_line_len); k++)
                tmp_text_id.push_back(text_id[k]);

            convert_batch_example(tokenizer, batchsize, seq_length, text_a, text_b, gt_classes, tmp_text_id, input_ids,
                                  segment_ids,
                                  input_mask, classes);

//            std::cout << "input_ids: " << std::endl;
//            for (int i = 0; i < batchsize * seq_length; i++) {
//                std::cout << input_ids[i] << " ";
//                if(i % seq_length == seq_length - 1)
//                    std::cout<< std::endl;
//            }
//            std::cout << std::endl;
//
//            std::cout << "segment_ids: " << std::endl;
//            for (int i = 0; i < batchsize * seq_length; i++) {
//                std::cout << segment_ids[i] << " ";
//                if(i % seq_length == seq_length - 1)
//                    std::cout<< std::endl;
//            }
//            std::cout << std::endl;
//
//            std::cout << "input_mask: " << std::endl;
//            for (int i = 0; i < batchsize * seq_length; i++) {
//                std::cout << input_mask[i] << " ";
//                if(i % seq_length == seq_length - 1)
//                    std::cout<< std::endl;
//            }
//            std::cout << std::endl;
//
//            std::cout << "label: " << std::endl;
//            for (int i = 0; i < batchsize; i++)
//                std::cout << classes[i] << " ";
//            std::cout << std::endl;

            float it_time;
            cudaEventRecord(start);
            int tmp_batchsize = min( (j+1) * batchsize, tot_line_len) - j * batchsize;
            model->handle->batchsize = tmp_batchsize;

            float loss = cuda_classify_train(
                    model,
                    input_ids,
                    segment_ids,
                    classes,
                    tmp_batchsize,
                    seq_length,
                    num_labels,
                    input_mask
            );

            now_loss += loss * tmp_batchsize;

//            std::cout << "temp batch loss: " << (*cpu_mem) << std::endl;
            if(j % 200 == 199) {
                std::cout << "outputRand: " << outputRand << std::endl;
                std::cout << "average loss: " << std::fixed << std::setprecision(10) << now_loss / min( (j+1) * batchsize, tot_line_len) << std::endl;
                outputRand++;
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&it_time, start, stop);
            total_time += it_time;
        }
        if(min_loss > now_loss)
            min_loss = now_loss;
        std::cout << "Round " << i + 1<< ": " << std::endl;
        std::cout << "***************  now_loss  *************" << std::endl;
        std::cout << "***************  " << std::fixed << std::setprecision(10) << now_loss / tot_line_len << "  *************" << std::endl;
        std::cout << "***************  min_loss  *************" << std::endl;
        std::cout << "***************  " << std::fixed << std::setprecision(10) << min_loss / tot_line_len << "  *************" << std::endl;
    }

    cubert_close_tokenizer(tokenizer);
    delete model;
}

void test_train(int batchsize, int seq_length, int nIter, bool base, int num_gpu) {
    bert *model = init_model(base, num_gpu, "", true);

    int test_word_id_seed[11] = {2040, 2001, 3958, 27227, 1029, 3958, 103,
                                 2001, 1037, 13997, 11510};
    int test_token_type_id_seed[11] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

    int attention_mask[11] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    int classes[8] = {1, 1, 1, 1, 1, 1, 1, 1};

    int *test_word_id, *test_token_type_id, *test_attention_mask;
    test_word_id = filling_inputs(test_word_id_seed, seq_length, 11, batchsize);
    test_token_type_id = filling_inputs(test_token_type_id_seed, seq_length, 11, batchsize);
    test_attention_mask = filling_inputs(attention_mask, seq_length, 11, batchsize);

    std::cout << "input_ids: " << std::endl;
    for (int i = 0; i < batchsize * seq_length; i++) {
        std::cout << test_word_id[i] << " ";
        if(i % seq_length == seq_length - 1)
            std::cout<< std::endl;
    }
    std::cout << std::endl;

    std::cout << "segment_ids: " << std::endl;
    for (int i = 0; i < batchsize * seq_length; i++) {
        std::cout << test_token_type_id[i] << " ";
        if(i % seq_length == seq_length - 1)
            std::cout<< std::endl;
    }
    std::cout << std::endl;

    std::cout << "input_mask: " << std::endl;
    for (int i = 0; i < batchsize * seq_length; i++) {
        std::cout << test_attention_mask[i] << " ";
        if(i % seq_length == seq_length - 1)
            std::cout<< std::endl;
    }
    std::cout << std::endl;

    std::cout << " Seq_length : " << seq_length << std::endl;
    std::cout << " Batchsize : " << batchsize << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *output_pinned;
    checkCudaErrors(cudaMallocHost((void **) &output_pinned,
                                   (1024) * model->handle->hidden_size * sizeof(float)));

    double total_time = 0;
    float learning_rate = 0.001;
    float learning_rate_decay = 0.99;
    for (int i = 0; i < nIter; i++) {
        printf("Round: %d\n", i);
        float it_time;
        cudaEventRecord(start);
        model->update_lr_start(learning_rate);
        float loss = cuda_classify_train(
                model,
                test_word_id,
                test_token_type_id,
                classes,
                batchsize,
                seq_length,
                2,
                test_attention_mask
        );
        std::cout << model->handle->learning_rate << std::endl;
        learning_rate *= learning_rate_decay;
        model->update_lr_end();

        printf("loss is %.10f\n", loss);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&it_time, start, stop);
        total_time += it_time;
    }

    delete model;

    double dSeconds = total_time / (double) nIter;
    printf("Time= %.2f(ms)\n", dSeconds);
}
}
