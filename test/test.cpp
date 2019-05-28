#include "../cuda_bert.cuh"
#include <string>

int main(int argc, char *argv[]){
   if (argc == 1) {
       test_train(1, 128, 300, false);
       test_train(1, 512, 300, false);
       test_train(2, 128, 300, false);
       test_train(2, 512, 300, false);
       test_train(4, 128, 300, false);
       test_train(4, 512, 300, false);
       test_train(8, 128, 300, false);
       test_train(8, 512, 300, false);
       test_train(16, 128, 300, false);
       test_train(32, 128, 300, false);
       
       test_train(1, 128, 300, true);
       test_train(1, 512, 300, true);
       test_train(2, 128, 300, true);
       test_train(2, 512, 300, true);
       test_train(4, 128, 300, true);
       test_train(8, 128, 300, true);
    //    bert_train(8, 3000, false, 0);
   }
   else{
       test_train(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), false, std::stoi(argv[4]));
   }
   return 0;
}
