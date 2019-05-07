#pragma once
#include "layer.hpp"
#include <iostream>

// __global__ void ForwardSoftmax(float *Z, int nColsZ, float *sumExp, float *A)
// {
//   int row = threadIdx.x;
//   int col = blockIdx.x;

//   atomicAdd(&sumExp[col], exp(Z[row * nColsZ + col]));

//   __syncthreads();

//   A[row * nColsZ + col] = exp(Z[row * nColsZ + col]) / sumExp[col];
// }

// __global__ void BackwardSoftmax(float *A, float *dA, int nColsdZ, float *dZ)
// {
//   int row = threadIdx.x;
//   int col = blockIdx.x;

//   dZ[row * nColsdZ + col] = dA[row * nColsdZ + col] * A[row * nColsdZ + col] *
//       (1 - A[row * nColsdZ + col]);
// }

class Softmax : public Layer
{
 public:
  Softmax()
  {
    /* Nothing to do here */
  }

  // CHECK WHY THE "&" after Matrix!!!!!!
  // CHECK WHY shared_ptr in matrix class!!


  // Matrix& Forward(Matrix& Z)
  // {
  //   this->Z = Z;
  //   // Z.CopyDeviceToHost();

  //   A.AllocateMemory(Z.nRows, Z.nCols);

  //   Matrix sumExp(1, Z.nCols);

  //   ForwardSoftmax<<<Z.nCols, Z.nRows>>>(Z.deviceMat.get(), Z.nCols,
  //       sumExp.deviceMat.get(), A.deviceMat.get());
  //   CheckErrors(cudaGetLastError(),
  //       "Softmax:: Kernel invocation: ForwardSoftmax");

  //   // Comment the below line if it's not needed on the host.
  //   // A.CopyDeviceToHost();

  //   return A;
  // }

  // Matrix& Backward(Matrix& dA, float lr)
  // {
  //   dZ.AllocateMemory(Z.nRows, Z.nCols);

  //   BackwardSoftmax<<<Z.nCols, Z.nRows>>>(A.deviceMat.get(), dA.deviceMat.get(),
  //       dZ.nCols, dZ.deviceMat.get());
  //   CheckErrors(cudaGetLastError(),
  //       "Softmax:: Kernel invocation: BackwardSoftmax");

  //   // Comment the below line if it's not needed on the host.
  //   // dZ.CopyDeviceToHost();

  //   return dZ;
  // }

 private:
  // Input and its derivative w.r.t. the loss.
  Matrix Z;
  Matrix dZ;

  // Output.
  Matrix A;
};

