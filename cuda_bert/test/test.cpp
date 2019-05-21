#include "../cuda_bert.cuh"
#include <string>

int main(int argc, char *argv[]){
   if (argc == 1){
      // test(1, 128, 300, false);
      // test(32, 128, 300, false);
      // test(1, 512, 300, false);
      // test(32, 512, 100, false);

      // test(1, 128, 10000, false);
      // test(1, 512, 10000, false);
      // test(2, 128, 10000, false);
      // test(2, 512, 10000, false);
      // test(4, 128, 10000, false);
      // test(4, 512, 10000, false);
      // test(8, 128, 10000, false);
      // test(8, 512, 10000, false);
      test(16, 128, 1000, false);
      test(16, 512, 1000, false);
      // test(32, 128, 1000, false);
      // test(32, 512, 1000, false);
      // test(64, 128, 1000, false);
      // test(64, 512, 1000, false);

      test(1, 128, 10000, true);
      test(1, 512, 10000, true);
      // test(2, 128, 10000, true);
      // test(2, 512, 10000, true);
      // test(4, 128, 10000, true);
      // test(4, 512, 10000, true);
      // test(8, 128, 10000, true);
      // test(8, 512, 10000, true);
      test(16, 128, 1000, true);
      test(16, 512, 1000, true);
      // test(32, 128, 1000, true);
      // test(32, 512, 1000, true);
      // test(64, 128, 1000, true);
      // test(64, 512, 1000, true);

   }

   else{
      test(std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]), false, std::stoi(argv[4]));
   }
   return 0;
}
