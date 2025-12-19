#pragma once
#include <vector>

using real_t = double;

struct Matrix {
    int rows, cols;
    std::vector<real_t> data;

    Matrix();
    Matrix(int r, int c, bool zero=true);

    real_t& operator()(int r, int c);
    const real_t& operator()(int r, int c) const;

    int size() const;
    void fill_rand(real_t a=-0.1, real_t b=0.1);
    Matrix transpose() const;
};

Matrix matmul(const Matrix &A, const Matrix &B);
Matrix add_rowwise(const Matrix &A, const Matrix &b);
