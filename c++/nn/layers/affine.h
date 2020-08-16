#ifndef NN_LAYER_AFFINE_H
#define NN_LAYER_AFFINE_H

// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../utils.h"
#include "layer.h"

namespace nn
{
namespace layers
{

struct AffineLayer : public Layer {
  Layer *prev_layer;
  std::vector<std::vector<float>> W;
  std::vector<float> b;

  AffineLayer(Layer *prev_layer_, size_t input_size, size_t output_size)
  {
    prev_layer = prev_layer_;
    W.assign(output_size, std::vector<float>(input_size, 0.0f));
    b.assign(output_size, 0.0f);
  }

  void load_file(std::string path)
  {
    std::ifstream ifs(path, std::ios::binary);
    W = nn::utils::read_2d(ifs, W.size(), W[0].size());
    b = nn::utils::read_1d(ifs, b.size());
  }

  virtual std::vector<float> forward()
  {
    const auto &x = prev_layer->forward();
    std::vector<float> y(W.size());
    for (std::size_t i = 0; i < W.size(); i++) {
      y[i] = nn::utils::multiply(W[i], x) + b[i];
    }
    return y;
  }
};

} // namespace layers
} // namespace nn

#endif // NN_LAYER_AFFINE_H
