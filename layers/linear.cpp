#pragma once
#include "layer.hpp"
#include <random>
#include <iostream>
#include "mpi.h"

class Linear : public Layer
{
 public:
  Linear(size_t inSize, size_t outSize) :
      W(outSize, inSize), b(outSize, 1)
  {
    dimBlockX = 16;
    dimBlockY = 16;
    InitializeParameters();
  }

  ~Linear(){}

  Matrix& ForwardCPU(Matrix& A)
  {
    if (A.nRows != W.nCols)
    {
        std::cerr << "ERROR: Numberrrrr of rows in the input matrix should be " <<
            "equal to the number of columns of the weight matrix." << std::endl;
    }

    this->A = A;
    Z.deAllocateMemoryP();
    Z.AllocateMemory(W.nRows, A.nCols);
    for (int i = 0; i < Z.nRows; i++)
    {
      for (int j = 0; j < Z.nCols; j++)
      {
        Z(i, j) = 0;
        for (int k = 0; k < A.nRows; k++)
        {
          Z(i, j) += W(i, k) * A(k, j);
        }

        Z(i, j) += b[i];
      }
    }
    return Z;
  }

  Matrix& BackwardCPU(Matrix& dZ, float lr = 0.01)
  {
      dA.deAllocateMemoryP();
      dA.AllocateMemory(A.nRows, A.nCols);

      for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        dA(i, j) = 0;
        for (int k = 0; k < W.nRows; k++)
        {
          dA(i, j) += W(k, i) * dZ(k, j);
        }
      }
    }

    UpdateParametersCPU(dZ, lr);
    return dA;
  }

  void UpdateParametersCPU(Matrix& dZ, float lr)
  {
    float dWValue;
    for (int i = 0; i < W.nRows; i++)
    {
      for (int j = 0; j < W.nCols; j++)
      {
        dWValue = 0;
        for (int k = 0; k < dZ.nCols; k++)
        {
          dWValue += dZ(i, k) * A(j, k);
        }
        W(i, j) -= lr * dWValue / dZ.nCols;
      }
    }

    float dbValue;
    for (int i = 0; i < dZ.nRows; i++)
    {
      dbValue = 0;
      for (int j = 0; j < dZ.nCols; j++)
      {
        dbValue += dZ(i, j);
      }
      b[i] -= lr * dbValue / dZ.nCols;
    }
  }

 private:
  // Parameters.
  Matrix W;
  Matrix b;

  // Input and its derivative w.r.t. the loss.
  Matrix A;
  Matrix dA;

  // Output.
  Matrix Z;

  int dimBlockX, dimBlockY;

  void InitializeParameters()
  {
    std::default_random_engine generator;
    std::normal_distribution<float> normalDist(0.0, 1.0);

    for (int i = 0; i < W.nRows; i++)
    {
      for (int j = 0; j < W.nCols; j++)
      {
        W(i, j) = normalDist(generator) * 0.01;

        // For testing, comment the above and uncomment the below line.
        // W(i, j) = 0.01;
      }
    }

    for (int i = 0; i < W.nRows; i++)
    {
      b[i] = 0;
    }
  }
};










// #include<stdio.h>
// #include<stdlib.h>
// // #include<mpi.h>
// #include<math.h>


// class Linear : public Layer
// {
//  public:
//   Linear(size_t inSize, size_t outSize) :
//       W(outSize, inSize), b(outSize, 1)
//   {
//     dimBlockX = 16;
//     dimBlockY = 16;
//     InitializeParameters();
//   }

//   ~Linear()
//   {
//     /* Nothing to do here */
//   }

//  Matrix& ForwardCPU(Matrix& A)
//   {
//     if (A.nRows != W.nCols)
//     {
//         std::cerr << "ERROR: Number of rows in the input matrix should be " <<
//             "equal to the number of columns of the weight matrix." << std::endl;
//     }

//     // A.CopyDeviceToHost();
//     this->A = A;

//     Z.AllocateMemory(W.nRows, A.nCols);

//     for (int i = 0; i < Z.nRows; i++)
//     {
//       for (int j = 0; j < Z.nCols; j++)
//       {
//         Z(i, j) = 0;
//         for (int k = 0; k < A.nRows; k++)
//         {
//           Z(i, j) += W(i, k) * A(k, j);
//         }

//         Z(i, j) += b[i];
//       }
//     }

//     // Z.CopyHostToDevice();

//     return Z;
//   }

// }
