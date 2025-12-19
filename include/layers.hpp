#pragma once
#include "matrix.hpp"

struct Dense {
    int in_dim, out_dim;

    Matrix W, b;
    Matrix mW, vW, mb, vb;  // Adam buffers
    Matrix x, z;

    Dense(int in_d, int out_d);

    Matrix forward(const Matrix &input, bool relu=true);
    Matrix backward(const Matrix &grad, bool had_relu,
                    Matrix &dW, Matrix &db);
};
