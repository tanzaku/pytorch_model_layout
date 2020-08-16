#ifndef NN_UTILS_H
#define NN_UTILS_H

// stdlib
#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace nn
{
namespace utils
{

float multiply(const std::vector<float> &u, const std::vector<float> &v)
{
  assert(u.size() == v.size());
  float res = 0;
  for (std::size_t i = 0; i < u.size(); i++) {
    res += u[i] * v[i];
  }
  return res;
}

// void from_file(FILE *fp, std::vector<float> &x)
// {
//   fread(x.data(), sizeof(x[0]), x.size(), fp);
// }

std::vector<float> read_1d(std::ifstream &ifs, std::size_t n)
{
  std::vector<float> x(n);
  ifs.read(reinterpret_cast<char *>(x.data()), sizeof(x[0]) * n);
  return x;
}

std::vector<float> read_1d(std::string path, std::size_t n)
{
  std::ifstream ifs(path, std::ios::binary);
  return read_1d(ifs, n);
}

std::vector<std::vector<float>> read_2d(std::ifstream &ifs, int row, int col)
{
  std::vector<std::vector<float>> x(row, std::vector<float>(col));
  for (auto &r : x) {
    ifs.read(reinterpret_cast<char *>(r.data()), sizeof(r[0]) * col);
  }
  return x;
}

std::vector<std::vector<float>> read_2d(std::string path, int row, int col)
{
  std::ifstream ifs(path, std::ios::binary);
  return read_2d(ifs, row, col);
}

} // namespace utils
} // namespace nn

#endif // NN_UTILS_H
