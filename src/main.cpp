#include <iostream>
#include <string>
#include "train.hpp"

int main(int argc, char** argv) {
    std::string csv = "train.csv";
    int epochs = 20;
    int batch = 64;
    double lr = 0.001;

    if (argc > 1) csv = argv[1];

    std::cout << "Training NN on MNIST CSV: " << csv << "\n";
    train_model(csv, epochs, batch, lr);
    return 0;
}
