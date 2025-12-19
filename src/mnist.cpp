#include "mnist.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

void load_mnist_csv(const std::string &path,
                    std::vector<std::vector<real_t>> &images,
                    std::vector<int> &labels,
                    int maxrows) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("CSV open failed");

    std::string line;
    getline(f, line);

    while (getline(f, line)) {
        if (maxrows > 0 && (int)labels.size() >= maxrows) break;

        std::stringstream ss(line);
        std::string tok;

        getline(ss, tok, ',');
        labels.push_back(std::stoi(tok));

        std::vector<real_t> img(784);
        for (int i=0;i<784;i++) {
            getline(ss, tok, ',');
            img[i] = std::stod(tok) / 255.0;
        }
        images.push_back(std::move(img));
    }
}

Matrix batch_to_matrix(const std::vector<std::vector<real_t>> &images,
                       int start, int batch) {
    int n = std::min(batch, (int)images.size()-start);
    Matrix X(n, 784);
    for (int i=0;i<n;i++)
        for (int j=0;j<784;j++)
            X(i,j) = images[start+i][j];
    return X;
}

std::vector<int> batch_labels(const std::vector<int> &labels,
                              int start, int batch) {
    int n = std::min(batch, (int)labels.size()-start);
    return std::vector<int>(labels.begin()+start,
                            labels.begin()+start+n);
}
