#pragma once
#include "layers.hpp"
#include <vector>
#include <string>

struct Model {
    std::vector<Dense> layers;

    void add(int in_dim, int out_dim);
    Matrix forward(const Matrix &X);

    void apply_adam(const std::vector<Matrix>& dW,
                    const std::vector<Matrix>& db,
                    real_t lr, int t);

    void save(const std::string &path);
    void load(const std::string &path);
};
