#pragma once

#include <memory>
#include <iostream>
// #include "error_handling.hpp"

/*
 * A Matrix class which can have the same matrix on both the CPU and GPU.
 * The functions like cudaMemcpy(in either direction), cudaMalloc, cudaFree
 * have been abstracted away in this class for convenience.
 */
class Matrix
{
 public:
  Matrix(){}

  Matrix(int nRows, int nCols) :
    nRows(nRows),
    nCols(nCols),
    memAllocated(false)
  {
    AllocateMemory();
  }

  void deAllocateMemoryP(){
      memAllocated = false;
  }

  void AllocateMemory()
  {
    if (!memAllocated)
    {
      // Allocate memory on the host.
      hostMat = std::shared_ptr<float>(new float[nRows * nCols],
          [&]/* lambda function. */(float* ptr){delete[] ptr;});

      memAllocated = true;
    }
  }

  void AllocateMemory(int nRows, int nCols)
  {
    if (!memAllocated)
    {
      this->nRows = nRows;
      this->nCols = nCols;

      AllocateMemory();
    }
  }

  float& operator[](const int index)
  {
    // Uncomment when debugging.
    // if (index >= nRows * nCols)
    // {
    //   std::cerr << "ERROR: Matrix::operator() index out of bounds" << std::endl;
    //   exit(-1);
    // }
    return hostMat.get()[index];
  }

  // Overload (row, col) for 2D indexing.
  float& operator()(const int row, const int col)
  {
    // Uncomment when debugging.
    // if (row >= nRows || col >= nCols)
    // {
    //   std::cerr << "ERROR: Matrix::operator() index out of bounds" << std::endl;
    //   exit(-1);
    // }

    return hostMat.get()[row * nCols + col];
  }

  std::shared_ptr<float> deviceMat;
  std::shared_ptr<float> hostMat;

  int nRows, nCols;

 private:
  bool memAllocated;
};
