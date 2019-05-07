#include "ffn.cpp"
#include "layers/linear.cpp"
#include "layers/layer.hpp"
#include "layers/relu.cpp"
#include "layers/sigmoid.cpp"
#include "layers/cross_entropy.cpp"
#include "utils.hpp"

int main()
{
  int nBatches = 200;
  int batchSize = 4096;
  Dataset dataset(batchSize, batchSize * nBatches);

  FFN network;
  network.Add(new Linear(2, 20));
  network.Add(new ReLU());
  network.Add(new Linear(20, 1));
  network.Add(new Sigmoid());

  network.Train(dataset /* dataset class */,
                0.1 /* learning rate */,
                10 /* epochs */,
                nBatches /* number of batchs */,
                true /* Whether to use CPU instead of GPU */);
    return 0;
}
