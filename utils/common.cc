#include "common.h"

void read_tsv(char *fname, std::vector <std::string> &items, std::vector <int> &gt_classes) {
    std::ifstream ifs(fname);
    if (ifs.fail()) {
        std::cerr << "error" << std::endl;
        return;
    }
    std::string line;
    while (getline(ifs, line)) {
        std::stringstream ss(line);
        std::string endString = "";
        std::string sentence = "";
        std::string tmp;
        while (getline(ss, tmp, '\t')) {
            if (endString != "")
                sentence = sentence + '\t' + endString;
            endString = tmp;
        }
        gt_classes.push_back(std::stoi(endString));
        items.push_back(sentence);
    }
    return;
}

template <typename T>
void debug_tensor(std::string tag, T* tensor, int max_x , int length_x, int max_y){
    /*
       print 1 to max_y lines from tensor:
       length_x: len(rows)
       max_x : len(rows) to print
    */
    std::cout<<" DEBUG TENSOR: ****  "<<tag<<"  *****  "<<std::endl;
    for(int i = 0; i < max_y; i++){
        for(int j = 0; j < max_x; j++)
            std::cout<<" "<<tensor[i*length_x + j]<<" ";
        std::cout<<std::endl;
    }
    std::cout<<" **************  DEBUG ENDING"<<std::endl;
}

template
void debug_tensor<float>(std::string tag, float* tensor, int max_x , int length_x, int max_y);

template <typename T>
void debug_tensor_gpu(std::string tag, void* gpu_tensor, int max_x, int length_x, int max_y){
    int length = length_x * max_y;
    T* cpu_mem;
    cpu_mem = (T *)malloc(sizeof(T) * length);
    T* tensor = static_cast<T *>(gpu_tensor);
    std::cout<<" \nDEBUG GPU TENSOR: ****  "<<tag<<"  ******  "<<std::endl;
    std::cout<<" Pointer:  --- "<<gpu_tensor<<" ------ "<<std::endl;
//    checkCudaErrors(cudaMemcpyAsync(cpu_mem, tensor, sizeof(T)*length, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cpu_mem, tensor, sizeof(T)*length, cudaMemcpyDeviceToHost));
    for(int i = 0; i < max_y; i++){
        for(int j = 0; j < max_x; j++)
            std::cout<<" "<<cpu_mem[i*length_x + j]<<" ";
        if(i*length_x + length_x -3 >= 0 && max_x + 3 <= length_x) {
            std::cout << " ... " << cpu_mem[i * length_x + length_x - 3] << "  "
                      << cpu_mem[i * length_x + length_x - 2] << "  "
                      << cpu_mem[i * length_x + length_x - 1] << "  ";
        }
        std::cout<<std::endl;
    }
    free(cpu_mem);
    std::cout<<" **************  DEBUG ENDING\n"<<std::endl;
}

template
void debug_tensor_gpu<float>(std::string tag, void* gpu_tensor, int max_x, int length_x, int max_y);

template
void debug_tensor_gpu<int>(std::string tag, void* gpu_tensor, int max_x, int length_x, int max_y);

template <typename T>
void dump_tensor(std::string file_name, 
                void* gpu_tensor,
                size_t dim1,
                size_t dim2,
                size_t dim3,
                size_t dim4,
                size_t dim5) {
    std::vector<size_t> shape;
    if(dim1 != 0) shape.push_back(dim1);
    if(dim2 != 0) shape.push_back(dim2);
    if(dim3 != 0) shape.push_back(dim3);
    if(dim4 != 0) shape.push_back(dim4);
    if(dim5 != 0) shape.push_back(dim5);
    long length = 1;
    for(auto dim : shape) length *= dim;
    T* cpu_mem;
    cpu_mem = (T *)malloc(sizeof(T) * length);
    T* tensor = static_cast<T *>(gpu_tensor);
    checkCudaErrors(cudaMemcpyAsync(cpu_mem, tensor, sizeof(T)*length, cudaMemcpyDeviceToHost));
    cnpy::npy_save("debug/cubert/" + file_name + ".npy", cpu_mem, shape);
    free(cpu_mem);
    return ;
}
