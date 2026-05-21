# micrograd.c

micrograd.c is a C implementation of a tiny autograd engine, inspired by Andrej Karpathy's micrograd.

More details on my website: <a href="https://andrearossetti.me/page.html?src=projects/micrograd.md" target="_blank">andrearossetti.me</a>

## Description

The autograd engine handles automatic differentiation, enabling the computation of gradients for tensor operations such as addition, multiplication, power, and ReLU activation. The neural network components include neurons, layers, and multi-layer perceptrons (MLPs) for building and training models.

## Training Loop

In the training loop, the code performs forward passes to compute the outputs and the loss, followed by a backward pass to compute gradients. The parameters are then updated using the Adam optimizer.

## Tests

The tests validate the functionality of the autograd engine and neural network components. They check the correctness of gradient computations, forward and backward passes, and overall network behavior.

## Makefile

The Makefile is used to compile the project. It defines rules for building the test executables.

### Usage

To build the project, run:

```sh
make
```

This will compile the source files and produce the test_engine and test_nn executables. You can then run these executables to test the autograd engine and neural network components.

