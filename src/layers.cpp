#include "layers.hpp"
#include <cmath>

Dense::Dense(int in_d, int out_d)
    : in_dim(in_d), out_dim(out_d),
      W(in_d, out_d), b(1, out_d),
      mW(in_d, out_d), vW(in_d, out_d),
      mb(1, out_d), vb(1, out_d) {
    W.fill_rand(-0.08, 0.08);
}

Matrix Dense::forward(const Matrix &input, bool relu) {
    x = input;
    z = add_rowwise(matmul(x, W), b);

    if (!relu) return z;

    Matrix a(z.rows, z.cols);
    for (int i=0;i<z.size();i++)
        a.data[i] = z.data[i] > 0 ? z.data[i] : 0.0;
    return a;
}

Matrix Dense::backward(const Matrix &grad, bool had_relu,
                       Matrix &dW, Matrix &db) {
    Matrix dz = grad;

    if (had_relu)
        for (int i=0;i<z.size();i++)
            dz.data[i] *= (z.data[i] > 0);

    dW = matmul(x.transpose(), dz);
    db = Matrix(1, dz.cols);

    for (int c=0;c<dz.cols;c++)
        for (int r=0;r<dz.rows;r++)
            db(0,c) += dz(r,c);

    real_t invb = 1.0 / dz.rows;
    for (auto &v : dW.data) v *= invb;
    for (auto &v : db.data) v *= invb;

    return matmul(dz, W.transpose());
}
