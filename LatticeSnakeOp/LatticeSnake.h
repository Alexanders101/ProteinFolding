//
// Created by alex on 7/17/18.
//

#ifndef LATTICESNAKEOP_LATTICESNAKE_H
#define LATTICESNAKEOP_LATTICESNAKE_H

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <map>

template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column- or
    // row-major). It will give you the same hash value for two different matrices if they
    // are the transpose of each other in different storage order.
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

#endif //LATTICESNAKEOP_LATTICESNAKE_H
