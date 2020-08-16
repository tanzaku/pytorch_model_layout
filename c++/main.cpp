
// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <limits>
#include <sstream>
#include <vector>

#include "nn/layers/affine.h"
#include "nn/layers/clipped_relu.h"
#include "nn/layers/input.h"

int main()
{
  nn::layers::InputLayer input_layer;
  auto affine_layer_1 = nn::layers::AffineLayer(&input_layer, 32, 16);
  auto relu_layer_1 = nn::layers::ClippedReLULayer(&affine_layer_1);
  auto affine_layer_2 = nn::layers::AffineLayer(&relu_layer_1, 16, 1);

  affine_layer_1.load_file("data//weight_row=16_col=32.bin");
  affine_layer_2.load_file("data/weight_row=1_col=16.bin");

  auto x = nn::utils::read_1d("data/input.bin", 32);
  input_layer.set(x);

  auto target = nn::utils::read_1d("data/target.bin", 1);
  auto output = nn::utils::read_1d("data/output.bin", 1);

  std::cerr << "target : " << target[0] << std::endl;
  std::cerr << "result : " << affine_layer_2.forward()[0] << std::endl;
  std::cerr << "pytorch output : " << output[0] << std::endl;

  return 0;
}
