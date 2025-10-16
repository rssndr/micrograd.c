// nn.c
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "nn.h"
#include "engine.h"

// Neuron functions
void neuron_zero_grad(Neuron *neuron) {
    for (int i = 0; i < neuron->n_inputs; i++) {
        neuron->w[i]->grad = 0;
    }
    neuron->b->grad = 0;
}

void neuron_init(Neuron *neuron, int n_inputs, NeuronConfig config) {
    // Guard against invalid input size
    if (n_inputs <= 0) {
        fprintf(stderr, "Error: n_inputs must be > 0\n");
        exit(EXIT_FAILURE);
    }

    neuron->w = (Value **)malloc(n_inputs * sizeof(Value*));
    if (!neuron->w) {
        fprintf(stderr, "Failed to allocate weights array\n");
        exit(EXIT_FAILURE);
    }

    neuron->n_inputs = n_inputs;
    neuron->config = config;

    // He initialization with validation
    for (int i = 0; i < n_inputs; i++) {
        neuron->w[i] = create_value(((double)rand() / RAND_MAX) * 2 - 1);
        if (!neuron->w[i]) {
            fprintf(stderr, "Failed to create weight Value\n");
            exit(EXIT_FAILURE);
        }
    }

    // Initialize bias safely
    neuron->b = create_value(0.0);
    if (!neuron->b) {
        fprintf(stderr, "Failed to create bias Value\n");
        exit(EXIT_FAILURE);
    }
}

Value* neuron_call(Neuron *neuron, Value **x) {
    Value *act = NULL;
    
    // First compute all products and keep them
    Value **products = malloc(neuron->n_inputs * sizeof(Value*));
    for (int i = 0; i < neuron->n_inputs; i++) {
        products[i] = mul(neuron->w[i], x[i]);
    }
    
    // Sum all products while maintaining connections
    act = products[0];
    for (int i = 1; i < neuron->n_inputs; i++) {
        act = add(act, products[i]);
    }
    
    // Add bias (maintaining connection)
    act = add(act, neuron->b);
    
    // Apply ReLU if needed (maintaining connection)
    if (neuron->config.nonlin == 1) {
        act = relu(act);
    }
    
    free(products);  // Only free the array, not the Values
    return act;
}

Value** neuron_parameters(Neuron *neuron) {
    int n_params = neuron->n_inputs + 1;
    Value **params = (Value **)malloc(n_params * sizeof(Value*));
    if (params == NULL) return NULL;

    // Return pointers to the actual parameters
    for (int i = 0; i < neuron->n_inputs; i++) {
        params[i] = neuron->w[i];  // Direct pointer to the weight Value
    }
    params[neuron->n_inputs] = neuron->b;  // Direct pointer to the bias Value

    return params;
}

void neuron_free(Neuron *neuron) {
    for (int i = 0; i < neuron->n_inputs; i++) {
        free(neuron->w[i]);  // Free each weight Value
    }
    free(neuron->w);  // Free the array of pointers
    free(neuron->b);  // Free the bias Value
}

// Layer functions
void layer_zero_grad(Layer *layer) {
    for (int i = 0; i < layer->n_neurons; i++) {
        neuron_zero_grad(&layer->neurons[i]);
    }
}

void layer_init(Layer *layer, int n_inputs, int n_neurons, NeuronConfig config) {
    layer->neurons = (Neuron *)malloc(n_neurons * sizeof(Neuron));
    layer->n_neurons = n_neurons;
    for (int i = 0; i < n_neurons; i++) {
        neuron_init(&layer->neurons[i], n_inputs, config);
    }
}

Value** layer_call(Layer *layer, Value **x) {
    Value **out = (Value**)malloc(layer->n_neurons * sizeof(Value*));
    if (out == NULL) return NULL;

    for (int i = 0; i < layer->n_neurons; i++) {
        out[i] = neuron_call(&layer->neurons[i], x);
    }
    return out;
}

Value** layer_parameters(Layer *layer) {
    int params_per_neuron = layer->neurons[0].n_inputs + 1;
    int n_params = layer->n_neurons * params_per_neuron;

    Value** params = (Value**)malloc(n_params * sizeof(Value*));
    if (params == NULL) return NULL;

    int pi = 0;
    for (int i = 0; i < layer->n_neurons; i++) {
        Value** neuron_params = neuron_parameters(&layer->neurons[i]);
        for (int j = 0; j < params_per_neuron; j++) {
            params[pi++] = neuron_params[j];
        }
        free(neuron_params);
    }

    return params;
}

void layer_free(Layer *layer) {
    for (int i = 0; i < layer->n_neurons; i++) {
        neuron_free(&layer->neurons[i]);
    }
    free(layer->neurons);
}

// MLP functions
void mlp_zero_grad(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        layer_zero_grad(&mlp->layers[i]);
    }
}

void mlp_init(MLP *mlp, int nin, int *nouts, int nouts_len) {
    int *sizes = (int*)malloc((nouts_len + 1) * sizeof(int));
    sizes[0] = nin;

    for (int i = 0; i < nouts_len; i++) {
        sizes[i+1] = nouts[i];
    }
    mlp->layers = (Layer*)malloc(nouts_len * sizeof(Layer));
    mlp->n_layers = nouts_len;
    
    for (int i = 0; i < nouts_len; i++) {
        NeuronConfig config = {.nonlin = (i != nouts_len - 1)};
        layer_init(&mlp->layers[i], sizes[i], sizes[i+1], config);
    }

    free(sizes);
}

Value** mlp_call(MLP *mlp, Value **x) {
    Value **current_x = x;

    for (int i = 0; i < mlp->n_layers; i++) {
        Value **layer_out = layer_call(&mlp->layers[i], current_x);

        if (i > 0) {
            free(current_x);
        }

        current_x = layer_out;
    }

    return current_x;
}

int mlp_n_params(MLP *mlp) {
    int n_params = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        Layer *layer = &mlp->layers[i];
        n_params += layer->n_neurons * (layer->neurons[0].n_inputs + 1); // weights + biases
    }
    return n_params;
}

Value** mlp_parameters(MLP *mlp) {
    int params_per_neuron = mlp->layers[0].neurons[0].n_inputs + 1;
    int n_params = mlp_n_params(mlp);
    Value** params = (Value**)malloc(n_params * sizeof(Value*));
    if (params == NULL) {
        fprintf(stderr, "Failed to allocate params array\n");
        exit(EXIT_FAILURE);
    }
    int pi = 0;
    for (int i = 0; i < mlp->n_layers; i++) {
        Layer *layer = &mlp->layers[i];
        Value** layer_params = layer_parameters(layer);
        for (int j = 0; j < layer->n_neurons * params_per_neuron; j++) {
            if (layer_params[j] == NULL) {
                fprintf(stderr, "NULL parameter at layer %d, index %d\n", i, j);
                exit(EXIT_FAILURE);
            }
            params[pi++] = layer_params[j];
        }
        free(layer_params);
    }
    return params;
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->n_layers; i++) {
        layer_free(&mlp->layers[i]);
    }
    free(mlp->layers);
}

