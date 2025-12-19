#include "matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

static std::mt19937 rng(123456);

Matrix::Matrix(): rows(0), cols(0) {}

Matrix::Matrix(int r, int c, bool zero): rows(r), cols(c), data(r*c) {
    if (zero) std::fill(data.begin(), data.end(), 0.0);
}

real_t& Matrix::operator()(int r, int c) {
    return data[r*cols + c];
}

const real_t& Matrix::operator()(int r, int c) const {
    return data[r*cols + c];
}

int Matrix::size() const {
    return rows * cols;
}

void Matrix::fill_rand(real_t a, real_t b) {
    std::uniform_real_distribution<real_t> dist(a, b);
    for (auto &v : data) v = dist(rng);
}

Matrix Matrix::transpose() const {
    Matrix t(cols, rows);
    for (int r=0; r<rows; r++)
        for (int c=0; c<cols; c++)
            t(c,r) = (*this)(r,c);
    return t;
}

Matrix matmul(const Matrix &A, const Matrix &B) {
    if (A.cols != B.rows)
        throw std::runtime_error("matmul shape mismatch");

    Matrix C(A.rows, B.cols);
    for (int i=0;i<A.rows;i++)
        for (int k=0;k<A.cols;k++) {
            real_t a = A(i,k);
            for (int j=0;j<B.cols;j++)
                C(i,j) += a * B(k,j);
        }
    return C;
}

Matrix add_rowwise(const Matrix &A, const Matrix &b) {
    if (b.rows != 1 || A.cols != b.cols)
        throw std::runtime_error("add_rowwise mismatch");

    Matrix R = A;
    for (int i=0;i<A.rows;i++)
        for (int j=0;j<A.cols;j++)
            R(i,j) += b(0,j);
    return R;
}
