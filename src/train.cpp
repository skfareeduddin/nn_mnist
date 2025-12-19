#include "model.hpp"
#include "mnist.hpp"
#include "train.hpp"

#include <iostream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <fstream>

static std::mt19937 rng(123456);

Matrix softmax(const Matrix &Z) {
    Matrix P(Z.rows, Z.cols);
    for (int i = 0; i < Z.rows; i++) {
        real_t maxv = -1e18;
        for (int j = 0; j < Z.cols; j++)
            maxv = std::max(maxv, Z(i, j));

        real_t sum = 0;
        for (int j = 0; j < Z.cols; j++) {
            P(i, j) = std::exp(Z(i, j) - maxv);
            sum += P(i, j);
        }
        for (int j = 0; j < Z.cols; j++)
            P(i, j) /= sum;
    }
    return P;
}

real_t cross_entropy(const Matrix &P, const std::vector<int> &y) {
    real_t loss = 0;
    for (int i = 0; i < P.rows; i++) {
        real_t p = std::max((real_t)1e-12, P(i, y[i]));
        loss -= std::log(p);
    }
    return loss / P.rows;
}

Matrix grad_softmax_ce(const Matrix &P, const std::vector<int> &y) {
    Matrix G = P;
    for (int i = 0; i < G.rows; i++)
        G(i, y[i]) -= 1.0;

    real_t invb = 1.0 / G.rows;
    for (auto &v : G.data) v *= invb;
    return G;
}

real_t accuracy(Model &model,
                 const std::vector<std::vector<real_t>> &X,
                 const std::vector<int> &y,
                 int max_eval = 2000) {
    int N = std::min((int)X.size(), max_eval);
    int correct = 0;

    for (int i = 0; i < N; i++) {
        Matrix Xi(1, 784);
        for (int j = 0; j < 784; j++) Xi(0, j) = X[i][j];

        Matrix logits = model.forward(Xi);
        Matrix P = softmax(logits);

        int pred = 0;
        for (int c = 1; c < 10; c++)
            if (P(0, c) > P(0, pred)) pred = c;

        if (pred == y[i]) correct++;
    }
    return (real_t)correct / N;
}

void train_model(const std::string &csv_path,
                 int epochs,
                 int batch_size,
                 double lr) {

    std::vector<std::vector<real_t>> images;
    std::vector<int> labels;

    load_mnist_csv(csv_path, images, labels);
    int N = images.size();

    int train_size = (int)(0.9 * N);

    std::vector<std::vector<real_t>> Xtrain(images.begin(), images.begin() + train_size);
    std::vector<int> ytrain(labels.begin(), labels.begin() + train_size);

    std::vector<std::vector<real_t>> Xval(images.begin() + train_size, images.end());
    std::vector<int> yval(labels.begin() + train_size, labels.end());

    Model model;
    model.add(784, 128);
    model.add(128, 64);
    model.add(64, 10);

    int t = 1;

    std::ofstream("loss_log.txt").close();

    for (int epoch = 1; epoch <= epochs; epoch++) {
        std::vector<int> idx(Xtrain.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);

        real_t epoch_loss = 0;
        int steps = Xtrain.size() / batch_size;

        for (int s = 0; s < steps; s++) {
            int start = s * batch_size;

            Matrix Xb(batch_size, 784);
            std::vector<int> yb(batch_size);

            for (int i = 0; i < batch_size; i++) {
                int id = idx[start + i];
                yb[i] = ytrain[id];
                for (int j = 0; j < 784; j++)
                    Xb(i, j) = Xtrain[id][j];
            }

            Matrix logits = model.forward(Xb);
            Matrix P = softmax(logits);

            epoch_loss += cross_entropy(P, yb);

            Matrix grad = grad_softmax_ce(P, yb);

            std::vector<Matrix> dW(model.layers.size());
            std::vector<Matrix> db(model.layers.size());

            for (int l = model.layers.size() - 1; l >= 0; l--) {
                bool had_relu = (l + 1 < model.layers.size());
                grad = model.layers[l].backward(grad, had_relu, dW[l], db[l]);
            }

            model.apply_adam(dW, db, lr, t++);
        }

        epoch_loss /= steps;
        real_t val_acc = accuracy(model, Xval, yval);

        std::ofstream flog("loss_log.txt", std::ios::app);
        flog << epoch << "," << epoch_loss << "\n";
        flog.close();

        std::cout << "Epoch " << epoch
                  << " | loss=" << epoch_loss
                  << " | val_acc=" << val_acc << "\n";
    }

    model.save("model.bin");
    std::cout << "Model saved to model.bin\n";
}
