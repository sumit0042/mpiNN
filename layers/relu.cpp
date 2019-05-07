#pragma once
#include "layer.hpp"
#include <iostream>

class ReLU : public Layer
{
 public:
  ReLU()
  {
    dimBlock = 64;
  }

  ~ReLU(){}

  Matrix& ForwardCPU(Matrix& Z)
  {
    this->Z = Z;

     A.AllocateMemory(Z.nRows, Z.nCols);

    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        if (Z(i, j) >= 0)
          A(i, j) = Z(i, j);
        else
          A(i, j) = 0;
      }
    }

    return A;
  }

  Matrix& BackwardCPU(Matrix& dA, float lr)
  {
      dZ.deAllocateMemoryP();
      dZ.AllocateMemory(Z.nRows, Z.nCols);

      for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        if (Z(i, j) >= 0)
          dZ(i, j) = dA(i, j);
        else
          dZ(i, j) = 0;
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

