#pragma once
#include <vector>
#include <string>
#include "matrix.hpp"

void load_mnist_csv(const std::string &path,
                    std::vector<std::vector<real_t>> &images,
                    std::vector<int> &labels,
                    int maxrows=-1);

Matrix batch_to_matrix(const std::vector<std::vector<real_t>> &images,
                       int start, int batch);

std::vector<int> batch_labels(const std::vector<int> &labels,
                              int start, int batch);
