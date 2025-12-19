# Neural Network from Scratch in C++

## Overview
This project is a fully functional implementation of a feedforward neural network written entirely from scratch in C++. It includes manual implementations of matrix operations, forward and backward propagation, ReLU activation, softmax with cross-entropy loss, and the Adam optimizer. The network is trained and evaluated on the MNIST handwritten digit dataset using CSV input.

The goal of this project is to demonstrate a deep understanding of machine learning fundamentals and low-level systems implementation without relying on external ML libraries.

---

## Features
- Fully custom matrix class with row-major storage
- Dense (fully connected) layers with ReLU activation
- Softmax output layer with cross-entropy loss
- Backpropagation implemented manually
- Adam optimizer implemented from scratch
- Mini-batch training with data shuffling
- Model serialization (save and load)
- Training loss logging and visualization
- Clean multi-file C++ project structure
- Built using CMake

---

## Model Architecture
- Input dimension: 784  
- Hidden layer 1: 128 neurons (ReLU)  
- Hidden layer 2: 64 neurons (ReLU)  
- Output layer: 10 neurons (Softmax)

---

## Dataset
The project uses the MNIST dataset in CSV format.

- Source: Kaggle Digit Recognizer  
- File: train.csv  
- Each row consists of one label followed by 784 pixel values  
- Pixel values are normalized to the range [0, 1]

---

## Build Instructions

### Prerequisites
- C++17 compatible compiler (GCC recommended)
- CMake version 3.10 or higher
- Python 3.x (for plotting)
- matplotlib Python package

### Build
From the project root directory:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Runnning Training
From the build directory:

```bash
./nn_mnist.exe ../train.csv
```

Training progress is printed after each epoch.
After training completes, the following files are generated:
- model.bin: serialized trained model
- loss_log.txt: training loss per epoch

---

## Plotting Training Loss

From the project root directory:
```bash
python plot_loss.py --log loss_log.txt --out loss_curve.png
```
This generates a plot showing training loss versus epoch.

---

## Results

- Validation accuracy: approximately 97 to 98 percent
- Loss decreases smoothly across epochs
- Stable training behavior using Adam optimizer

---

## Project Structure

```
nn_mnist/
├── CMakeLists.txt
├── include/
│   ├── matrix.hpp
│   ├── layers.hpp
│   ├── model.hpp
│   ├── mnist.hpp
│   └── train.hpp
├── src/
│   ├── matrix.cpp
│   ├── layers.cpp
│   ├── model.cpp
│   ├── mnist.cpp
│   ├── train.cpp
│   └── main.cpp
├── loss_curve.png
├── plot_loss.py
└── README.md
```

---

## Outcomes
- Implemented neural networks without high-level frameworks
- Gained deep understanding of backpropagation and optimization
- Practiced low-level numerical programming in C++
- Built a clean and maintainable multi-file C++ project
- Learned to debug real-world compilation and linking issues
