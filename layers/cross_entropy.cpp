#pragma once
#include "../matrix.cpp"
#include <iostream>
#include <math.h>


class CrossEntropy
{
 public:
  CrossEntropy(){}

  ~CrossEntropy(){}

  float Forward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: NumberCrossEntropyFwd of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    float* loss;
    *loss = 0.0f;

    for (int col = 0; col < output.nCols; ++col)
    {
//      float temp = -(labels[col] * logf(output[col]) + logf(1 - output[col])
//      * (1 - labels[col]));
      float temp = fabsf(labels[col]-output[col]);
      *loss += temp;
    }
    
    lossReturn = *loss;
    
    return lossReturn / output.nCols;
  }

  Matrix& Backward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: Numberrr of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    dOutput.AllocateMemory(output.nRows, output.nCols);

    for (int col = 0; col < output.nCols; ++col)
    {
      dOutput[col] = (labels[col] / output[col] - (1 - labels[col]) /
          (1 - output[col])) * -1;
    }

    return dOutput;
  }

 private:
  Matrix dOutput;

  float lossReturn;
};
