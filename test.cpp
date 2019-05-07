#include "matrix.cpp"
#include <iostream>
#include "layers/linear.cpp"

using namespace std;

int maint()
{


  // Dataset d(64, 64 * 10);
  // Matrix a = d.LabelBatches().at(5);
  // for (int i = 0; i < a.nRows; i++)
  // {
  //   for (int j = 0; j < a.nCols; j++)
  //   {
  //     cout << a(i, j) << " ";
  //   }
  //   cout << endl;
  // }
  Matrix a(3, 3);
  a(0, 0) = 3;
  a(1, 0) = 25;
  a(2, 0) = 45;
  a(0, 1) = 3;
  a(1, 1) = 35;
  a(2, 1) = 35;
  a(0, 2) = 2;
  a(1, 2) = 4;
  a(2, 2) = 4;
  // a.CopyHostToDevice();

  std::cout<<a(2,1)<<std::endl;

  Linear l(3, 3);
  Matrix g = l.BackwardCPU(a);
  // g.CopyDeviceToHost();
  for (int i = 0; i < g.nRows; i++)
  {
    for (int j = 0; j < g.nCols; j++)
    {
      cout << g(i, j) << " ";
    }
    cout << endl;
  }

    return 0;
}
