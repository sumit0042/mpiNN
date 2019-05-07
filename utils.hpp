//
// Created by root on 6/5/19.
//

#ifndef CUDACPU_UTILS_HPP
#define CUDACPU_UTILS_HPP

#include <vector>
#include "matrix.cpp"

float Accuracy(Matrix& output, Matrix& labels);
class Dataset{
public:
    Dataset(int batchSize, int nPoints);
//    ~Dataset();
    std::vector<Matrix>& DataBatches(){
        return dataBatches;
    }
    std::vector<Matrix>& LabelBatches(){
        return labelBatches;
    }
private:
    int batchSize;
    std::vector<Matrix> dataBatches;
    std::vector<Matrix> labelBatches;
};

#endif //CUDACPU_UTILS_HPP
