#ifndef NN_LAYER_CLIPPED_RELU_H
#define NN_LAYER_CLIPPED_RELU_H

// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "../utils.h"
#include "layer.h"

namespace nn
{
namespace layers
{

struct ClippedReLULayer : public Layer {
  Layer *prev_layer;

  ClippedReLULayer(Layer *prev_layer_)
  {
    prev_layer = prev_layer_;
  }

  virtual std::vector<float> forward()
  {
    const auto &x = prev_layer->forward();
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); i++) {
      // y[i] = std::max(0.0f, std::min(128.0f, x[i] / nn::consts::kScaleReLU));
      y[i] = std::max(0.0f, std::min(5.0f, x[i])) * 10.0f;
    }
    return y;
  }
};

} // namespace layers
} // namespace nn

#endif // NN_LAYER_CLIPPED_RELU_H
