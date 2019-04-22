CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
project(cudaBERT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cnpy/)

add_subdirectory(cnpy)

find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found.")
endif()

add_definitions(-DHAVE_CUDA)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/)

message("warning: helper headers locate at ")
message(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc/)

list(APPEND cudaBERT_EXTERNAL_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-Xptxas -O3;-Xcompiler -O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler -fPIC)
set(CUDA_SEPARABLE_COMPILATION ON)

set(SOURCE_FILES
        utils/bert.cu
        utils/common.cc
        utils/load_model.cc
        utils/manager.cu
        ops/elementwise.cu
        ops/embedding.cu
        ops/layernorm.cu
        ops/linear.cu
        ops/matmul.cu
        ops/softmax.cu
        ops/batch_matmul.cu
        cuda_bert.cu)

cuda_add_library(cudaBERT SHARED ${SOURCE_FILES})
add_dependencies(cudaBERT cnpy)
target_link_libraries(cudaBERT cnpy ${cudaBERT_EXTERNAL_LIBRARIES})

cuda_add_executable(unit_test test/test.cpp)
target_link_libraries(unit_test cudaBERT)
