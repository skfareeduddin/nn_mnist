#pragma once
#include <string>

void train_model(const std::string &csv_path,
                 int epochs,
                 int batch_size,
                 double lr);
