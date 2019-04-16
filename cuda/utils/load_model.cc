#include "load_model.h"

void GetFileNames(std::string path, std::vector<std::string>& filenames) {
    DIR *pDir;
    struct dirent* ptr;
    if(!(pDir = opendir(path.c_str())))
        return;
    while((ptr = readdir(pDir))!=0) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
            filenames.push_back(path + "/" + ptr->d_name);
    }
    closedir(pDir);
}

tagged_tensor* look_up_tts(std::vector<tagged_tensor *> tts, std::vector<std::string> keys) {
    for( auto tt : tts){
        int flag = 1;
        for( auto key : keys){
            std::size_t find = tt->name.find(key);
            if (find == std::string::npos){
                flag = 0;
                //std::cout<<key<<"  "<<tt->name<<std::endl;
                break;
            }
        }
        if( flag == 1 ){
            return tt;
        }
    }
    std::cout<<"Error: No keys in tts : ";
    for(auto key : keys){
        std::cout<<" "<<key<<" ";
    }
    std::cout<<std::endl;
    return nullptr;
}

dict_weights load_dict_weights(std::vector<tagged_tensor *> tts, int num_layer) {
    std::vector<std::string> num_layers;
    num_layers.resize(num_layer);
    for(int i = 0 ; i < num_layer; i++){
        num_layers[i] = "_" + std::to_string(i) + "_";
    }
    std::vector<std::string> weights_name = {
  //      "LayerNorm_beta",    Only Once by look_up_tts
  //      "LayerNorm_gamma",   Only Once by look_up_tts
  //       "pooler_dense_bias"
  //       "pooler_dense_kernel"
        "attention_output_LayerNorm_beta",
        "attention_output_LayerNorm_gamma",
        "output_LayerNorm_beta",
        "output_LayerNorm_gamma",
        "attention_output_dense_bias"   ,
        "attention_output_dense_kernel" ,
        "attention_self_key_bias"     ,
        "attention_self_key_kernel"   ,
        "attention_self_query_bias"   ,
        "attention_self_query_kernel" ,
        "attention_self_value_bias"   ,
        "attention_self_value_kernel" ,
        "intermediate_dense_kernel"   ,
        "intermediate_dense_bias"     ,
        "output_dense_bias"           ,
        "output_dense_kernel" 
    };

    dict_weights target;
    std::vector<float *> temp;

    for(auto name : weights_name){
        temp.clear();
        for(auto id : num_layers){
            std::vector<std::string> keys = {id + name};
            tagged_tensor* tt;
            tt = look_up_tts(tts, keys);
            temp.push_back(tt->gpu_mem);
        }
        target[name] = temp;
        //std::cout<<name<<" has num "<<temp.size()<<std::endl;
    }

    return target;
}

void load_from_dir_to_GPU(std::string model_path_dir, std::vector<tagged_tensor *>& tts) {
    std::vector<std::string> file_name;
    GetFileNames(model_path_dir, file_name);
    for(auto name : file_name){
        cnpy::NpyArray* arr = new cnpy::NpyArray; 
        *arr = cnpy::npy_load(name);      
        float* loaded = arr->data<float>();
        assert(arr->word_size == sizeof(float));
        tagged_tensor *tt = new tagged_tensor(name, arr->shape, loaded);
        float* gpu_mem;
        checkCudaErrors(cudaMalloc((void**)&gpu_mem, 
                            sizeof(float) * tt->num_elements));
        checkCudaErrors(cudaMemcpy(gpu_mem, tt->cpu_mem, 
                                       sizeof(float) * tt->num_elements,
                                       cudaMemcpyHostToDevice));
        //std::cout<<tt->debug_string()<<std::endl;   
        tt->gpu_mem = gpu_mem;
        tt->cpu_mem = nullptr;
        tts.push_back(tt);
        delete arr;
    }
    std::cout<<"Success: load npy from "<<model_path_dir<<" to GPU"<<std::endl;
}
