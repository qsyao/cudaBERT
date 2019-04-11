# Compile

nvcc cuda_bert.cu -o libcubert.so -lcublas -I ./reference/ -lcnpy -L ./ -lz --std=c++11 -shared -Xcompiler -fPIC