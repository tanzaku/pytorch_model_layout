#ifndef NN_LAYER_INPUT_H
#define NN_LAYER_INPUT_H

// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "layer.h"

namespace nn
{
namespace layers
{

struct InputLayer : public Layer {
  std::vector<float> x;

  void set(std::vector<float> x_)
  {
    x = x_;
  }

  virtual std::vector<float> forward()
  {
    return x;
  }
};

} // namespace layers
} // namespace nn

#endif // NN_LAYER_INPUT_H
