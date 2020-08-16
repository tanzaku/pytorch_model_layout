#ifndef NN_LAYER_H
#define NN_LAYER_H

// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "../utils.h"

namespace nn
{
namespace layers
{

struct Layer {
public:
  virtual std::vector<float> forward() = 0;
};

} // namespace layers
} // namespace nn

#endif // NN_LAYER_H
