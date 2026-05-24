# micrograd.c

A C implementation of a scalar-valued autograd engine and neural network library, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

More details on my website: <a href="https://andrearossetti.me/projects/micrograd" target="_blank">andrearossetti.me</a>

## What it does

Builds a computation graph of scalar `Value` nodes as operations are performed, then runs backpropagation through that graph to compute gradients automatically. On top of the engine, a simple MLP implementation allows you to define, train, and evaluate feedforward networks.

A 3→4→4→1 network trained with Adam on a toy 4-sample dataset reaches near-zero MSE loss within 200 epochs.

## Project Structure

```
.
├── engine.h / engine.c     # Value struct, operations (add, mul, relu, …), backward()
├── nn.h / nn.c             # Neuron, Layer, MLP
├── test_engine.c           # Unit tests for individual operations
├── test_nn.c               # MLP init, forward pass, backward pass, training loop
└── Makefile
```

## How to Run

```sh
# Build
make

# Run engine unit tests
./test_engine

# Run MLP tests + training
./test_nn
```

## Features

- Dynamic computation graph — built on the fly as operations execute
- Reverse-mode autodiff via topological sort and backpropagation
- Operations: `add`, `mul`, `power`, `relu` (leaky), `neg`, `sub`, `truediv`
- Leaky ReLU activation (leak = 0.01) to prevent dying neurons
- Configurable MLP: arbitrary layer sizes, nonlinearity per layer
- Adam optimiser with bias correction
- Memory-safe: graph walker respects parameter ownership, zero errors under Valgrind

## Demo
![](demo.gif)

## License

MIT
