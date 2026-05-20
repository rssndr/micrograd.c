// nn.h
#ifndef NN_H
#define NN_H

#include "engine.h"

typedef struct {
    int nonlin;            // Nonlinearity flag
    // Future parameters can be added here
} NeuronConfig;

typedef struct {
    Value **w;               // Array of weights
    Value *b;                // Bias
    int n_inputs;           // Number of inputs
    NeuronConfig config;    // Additional neuron configuration 
} Neuron;

typedef struct {
    Neuron *neurons;        // Array of neurons
    int n_neurons;          // Number of neurons in the layer
} Layer;

typedef struct {
    Layer *layers;          // Array of layers
    int n_layers;           // Number of layers in the MLP
} MLP;

typedef struct {
    Value ***layer_outputs;
    int n_layers;
} MLPOutput;

void neuron_zero_grad(Neuron *neuron);
void neuron_init(Neuron *neuron, int n_inputs, NeuronConfig config);
Value* neuron_call(Neuron *neuron, Value **x);
Value** neuron_parameters(Neuron *neuron);
void neuron_free(Neuron *neuron);

void layer_zero_grad(Layer *layer);
void layer_init(Layer *layer, int n_inputs, int n_neurons, NeuronConfig config);
Value** layer_call(Layer *layer, Value **x);
Value** layer_parameters(Layer *layer);
void layer_free(Layer *layer);

void mlp_zero_grad(MLP *mlp);
void mlp_init(MLP *mlp, int nin, int *nouts, int nouts_len);
Value** mlp_call(MLP *mlp, Value **x);
int mlp_n_params(MLP *mlp);
Value** mlp_parameters(MLP *mlp);
void mlp_free(MLP *mlp);

#endif

