#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <ctime>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <deque>
#include <stack>
#include <queue>
#include <bitset>
#include <algorithm>
#include <numeric>
#include <utility>
#include <sstream>

void read_tsv(char *fname) {
    std::ifstream ifs(fname);
    if (ifs.fail()) {
        std::cerr << "error" << std::endl;
        return;
    }
    std::string line;
    std::vector <std::string> items;
    std::vector <std::string> classes;
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
        classes.push_back(endString);
        items.push_back(sentence);
    }
    for (int i = 0; i < items.size(); i++) {
        std::cout << "[" << i << "]" << items[i] << std::endl;
        std::cout << "[" << i << "]" << classes[i] << std::endl;
    }
    return;
}

int main(int argc, char *argv[]) {
    read_tsv("/home/wenxh/zyc/bert_train/cuda_bert/data/deepqa_train_10w.tsv");
    return 0;
}