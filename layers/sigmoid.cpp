#pragma once
#include "layer.hpp"
#include <iostream>
#include <math.h>

class Sigmoid : public Layer
{
 public:
  Sigmoid()
  {
    dimBlock = 64;
  }

  Matrix& ForwardCPU(Matrix& Z)
  {
    this->Z = Z;
    A.deAllocateMemoryP();
    A.AllocateMemory(Z.nRows, Z.nCols);
    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        A(i, j) = 1 / (1 + exp(-Z(i, j)));
      }
    }
    return A;
  }

  Matrix& BackwardCPU(Matrix& dA, float lr)
  {
      dZ.deAllocateMemoryP();
    dZ.AllocateMemory(Z.nRows, Z.nCols);
    for (int i = 0; i < Z.nRows; i++)
    {
      for (int j = 0; j < Z.nCols; j++)
      {
        dZ(i, j) = 1 / (1 + exp(-Z(i, j))) * (1 - 1 / (1 + exp(-Z(i, j)))) *
          dA(i, j);
      }
    }
    return dZ;
  }

 private:
  // Input and its derivative w.r.t. the loss.
  Matrix Z;
  Matrix dZ;

  // Output.
  Matrix A;

  int dimBlock;
};
