#include "../cuda_bert.cuh"
#include <string>

int main(int argc, char *argv[]){
   if (argc == 1) {
       test_train(8, 128, 30, false);
   }
   else{
       test_train(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), false, std::stoi(argv[4]));
   }
   return 0;
}
