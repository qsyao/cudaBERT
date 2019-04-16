#ifndef LOAD_MODEL_CUDA_BERT
#define LOAD_MODEL_CUDA_BERT

#include <iostream>
#include <sys/types.h>
#include <string>

#include <dirent.h>
#include "cuda_runtime.h"

#include "common.h"

void GetFileNames(std::string path, std::vector<std::string>& filenames);

tagged_tensor* look_up_tts(std::vector<tagged_tensor *> tts, std::vector<std::string> keys);

dict_weights load_dict_weights(std::vector<tagged_tensor *> tts, int num_layer);

void load_from_dir_to_GPU(std::string model_path_dir, std::vector<tagged_tensor *>& tts);

#endif
