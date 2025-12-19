#include "model.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>


void Model::add(int in_dim, int out_dim) {
    layers.emplace_back(in_dim, out_dim);
}

Matrix Model::forward(const Matrix &X) {
    Matrix out = X;
    for (size_t i = 0; i < layers.size(); i++)
        out = layers[i].forward(out, i + 1 < layers.size());
    return out;
}

void Model::apply_adam(const std::vector<Matrix>& dW,
                       const std::vector<Matrix>& db,
                       real_t lr, int t) {

    const real_t b1 = 0.9;
    const real_t b2 = 0.999;
    const real_t eps = 1e-8;

    for (size_t i = 0; i < layers.size(); i++) {
        Dense &L = layers[i];

        for (int k = 0; k < L.W.size(); k++) {
            L.mW.data[k] = b1 * L.mW.data[k] + (1 - b1) * dW[i].data[k];
            L.vW.data[k] = b2 * L.vW.data[k] + (1 - b2) * dW[i].data[k] * dW[i].data[k];

            real_t mh = L.mW.data[k] / (1 - std::pow(b1, t));
            real_t vh = L.vW.data[k] / (1 - std::pow(b2, t));

            L.W.data[k] -= lr * mh / (std::sqrt(vh) + eps);
        }

        for (int k = 0; k < L.b.size(); k++) {
            L.mb.data[k] = b1 * L.mb.data[k] + (1 - b1) * db[i].data[k];
            L.vb.data[k] = b2 * L.vb.data[k] + (1 - b2) * db[i].data[k] * db[i].data[k];

            real_t mh = L.mb.data[k] / (1 - std::pow(b1, t));
            real_t vh = L.vb.data[k] / (1 - std::pow(b2, t));

            L.b.data[k] -= lr * mh / (std::sqrt(vh) + eps);
        }
    }
}

void Model::save(const std::string &path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open model file for saving");

    int num_layers = layers.size();
    out.write((char*)&num_layers, sizeof(int));

    for (const Dense &L : layers) {
        out.write((char*)&L.in_dim, sizeof(int));
        out.write((char*)&L.out_dim, sizeof(int));

        out.write((char*)L.W.data.data(), L.W.size() * sizeof(real_t));
        out.write((char*)L.b.data.data(), L.b.size() * sizeof(real_t));
    }
}

void Model::load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open model file for loading");

    int num_layers;
    in.read((char*)&num_layers, sizeof(int));

    layers.clear();
    layers.reserve(num_layers);

    for (int i = 0; i < num_layers; i++) {
        int in_dim, out_dim;
        in.read((char*)&in_dim, sizeof(int));
        in.read((char*)&out_dim, sizeof(int));

        layers.emplace_back(in_dim, out_dim);
        Dense &L = layers.back();

        in.read((char*)L.W.data.data(), L.W.size() * sizeof(real_t));
        in.read((char*)L.b.data.data(), L.b.size() * sizeof(real_t));
    }
}
