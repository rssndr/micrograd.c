#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

void free_value(Value* v) {
    if (v == NULL) return;
    static Value* visited[10000]; // Avoid double-free
    static int visited_size = 0;
    for (int i = 0; i < visited_size; i++) {
        if (visited[i] == v) return;
    }
    visited[visited_size++] = v;
    free_value(v->prev[0]);
    free_value(v->prev[1]);
    free(v);
    if (visited_size >= 10000) visited_size = 0; // Reset for next use
}

// Test MLP initialization and parameter count
void test_mlp_init() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network: 4 hidden, 4 hidden, 1 output
    mlp_init(&mlp, 3, nouts, 3);
    
    int expected_params = (3*4 + 4) + (4*4 + 4) + (4*1 + 1);  // (inputs*hidden + hidden) + (hidden*hidden + hidden) + (hidden*output + output)
    int actual_params = mlp_n_params(&mlp);
    printf("MLP Init Test:\n");
    printf("  Expected params: %d\n", expected_params);
    printf("  Actual params:   %d (%s)\n\n", actual_params, 
          actual_params == expected_params ? "PASS" : "FAIL");
    
    mlp_free(&mlp);
}

// Test forward pass dimensions
void test_forward_pass() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network: 4 hidden, 4 hidden, 1 output
    mlp_init(&mlp, 3, nouts, 3);
    
    // Create input Values
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    
    Value** outputs[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        printf("Forward Pass Test %d:\n", i+1);
        printf("  Expected output size: 1\n");
        printf("  Actual output size:   %d (%s)\n\n", mlp.layers[2].n_neurons, 
              mlp.layers[2].n_neurons == 1 ? "PASS" : "FAIL");
    }
    
    // Cleanup
    for (int i = 0; i < 4; i++) {
        free(outputs[i]);
        for (int j = 0; j < 3; j++) {
            free(inputs[i][j]);
        }
    }
    mlp_free(&mlp);
}

// Test gradient computation
void test_backward() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);
    
    // Create input and target
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    Value* targets[4] = {create_value(1.0), create_value(-1.0), create_value(-1.0), create_value(1.0)};
    
    // Forward pass
    Value** outputs[4];
    Value* losses[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        losses[i] = power(sub(outputs[i][0], targets[i]), 2.0);
    }
    
    // Backward pass
    mlp_zero_grad(&mlp);
    for (int i = 0; i < 4; i++) {
        backward(losses[i]);
    }
    
    // Check gradients
    Value** params = mlp_parameters(&mlp);
    int all_zero = 1;
    for(int i=0; i<mlp_n_params(&mlp); i++) {
        if(params[i]->grad != 0.0) {
            all_zero = 0;
            break;
        }
    }
    
    printf("Backward Pass Test:\n");
    printf("  All params have non-zero grad: %s (%s)\n\n", 
          all_zero ? "NO" : "YES", 
          all_zero ? "FAIL" : "PASS");
    
    // Cleanup
    free(params);
    for (int i = 0; i < 4; i++) {
        free(outputs[i]);
        free(losses[i]);
        for (int j = 0; j < 3; j++) {
            free(inputs[i][j]);
        }
        free(targets[i]);
    }
    mlp_free(&mlp);
}

// Simple training test
void test_training() {
    MLP mlp;
    int nouts[] = {4, 4, 1};  // 3-layer network
    mlp_init(&mlp, 3, nouts, 3);
    
    // Inputs and targets
    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0), create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0), create_value(1.0)},
        {create_value(1.0), create_value(1.0), create_value(-1.0)}
    };
    Value* targets[4] = {create_value(1.0), create_value(-1.0), create_value(-1.0), create_value(1.0)};

    // Get all parameters once (outside the loop)
    Value** params = mlp_parameters(&mlp);
    int n_params = mlp_n_params(&mlp);

    // Adam state variables
    double* m = calloc(n_params, sizeof(double));  // 1st moment
    double* v = calloc(n_params, sizeof(double));  // 2nd moment
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    double lr = 0.02;  // Adam typically uses smaller learning rates

    // Training loop
    float total_losses[200];
    for(int epoch=0; epoch<200; epoch++) {
        mlp_zero_grad(&mlp);
        
        // Forward pass and accumulate loss
        Value* total_loss = create_value(0.0);
        Value* losses[4];
        for (int i = 0; i < 4; i++) {
            Value** output = mlp_call(&mlp, inputs[i]);
            losses[i] = power(sub(output[0], targets[i]), 2.0);  // L2 loss
            total_loss = add(total_loss, losses[i]);
            free(output);  // Free output array (not Values)
        }
        
        // Compute mean loss
        Value* divisor = create_value(4.0);  // Batch size = 4
        Value* avg_loss = truediv(total_loss, divisor);
        
        // Backward pass
        backward(avg_loss);
        
        // Update weights (SGD)
        for(int i=0; i<mlp_n_params(&mlp); i++) {
            if (params[i] == NULL) {
                fprintf(stderr, "NULL param at index %d\n", i);
                exit(EXIT_FAILURE);
            }
            // Update moments
            m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * pow(params[i]->grad, 2);

            // Bias correction
            double m_hat = m[i] / (1 - pow(beta1, epoch + 1));
            double v_hat = v[i] / (1 - pow(beta2, epoch + 1));

            // Update parameter
            params[i]->data -= lr * m_hat / (sqrt(v_hat) + eps);
        }

        // Store and print loss
        total_losses[epoch] = avg_loss->data;
        printf("Training Step %d - Average Loss: %.8f\n", epoch+1, avg_loss->data);
        
        // Cleanup
        free_value(avg_loss); // Frees avg_loss, total_loss, losses, and their graphs
        free_value(divisor);
        for (int i = 0; i < 4; i++) {
            free_value(losses[i]->prev[0]); // Free sub(output[0], targets[i])
            free_value(losses[i]); // Already freed via avg_loss, but ensure no double-free
        }
    }

    // After training loop: Evaluate final predictions and loss
    printf("\nFinal Training Results:\n");
    Value* final_loss = create_value(0.0);
    for (int i = 0; i < 4; i++) {
        Value** output = mlp_call(&mlp, inputs[i]);
        Value* loss = power(sub(output[0], targets[i]), 2.0);  // L2 loss
        final_loss = add(final_loss, loss);

        // Print predicted vs actual values
        printf("Input %d: Predicted = %.8f, Actual = %.8f\n", i+1, output[0]->data, targets[i]->data);

        free_value(loss);
        free(output);  // Free output array (not Values)
    }
    Value* final_avg_loss = truediv(final_loss, create_value(4.0));
    printf("Final Average Loss: %.8f\n", final_avg_loss->data);

    // Cleanup
    free_value(final_avg_loss);
    free(m);
    free(v);
    free(params);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) free(inputs[i][j]);
        free(targets[i]);
    }
    mlp_free(&mlp);
}

int main() {
    srand(time(NULL));
    
    test_mlp_init();
    test_forward_pass();
    test_backward();
    test_training();
    
    return 0;
}

