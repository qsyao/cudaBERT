#include "./reference/cnpy.h"
#include <vector>

int main()
{
    float *weights;
    float *bias;
    std::vector<size_t> weights_shape;
    std::vector<size_t> bias_shape;
    weights_shape.push_back(768);
    weights_shape.push_back(2);
    bias_shape.push_back(2);
    int length = 2 * 768;
    weights = (float *)malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++)
        weights[i] = 1.0;
    cnpy::npy_save("model_npy/base_uncased/classifier_weight.npy", weights, weights_shape);
    length = 2;
    bias = (float *)malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++)
        bias[i] = 0.0;
    cnpy::npy_save("model_npy/base_uncased/classifier_bias.npy", bias, bias_shape);
    return 0;
}

// nvcc createSequenceClassificationData.cu -o test -lcnpy -L ./ --std=c++11
