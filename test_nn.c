#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "nn.h"

/* ============================================================
 * Graph memory management
 * ============================================================
 *
 * OWNERSHIP MODEL
 * ───────────────
 * There are three categories of Value nodes in a computation graph:
 *
 *   1. MLP PARAMETERS (weights & biases): owned by the MLP struct,
 *      allocated in neuron_init(), freed in mlp_free().  Must never
 *      be touched by test-level cleanup code.
 *
 *   2. INPUT / TARGET leaf Values: allocated by the test, passed into
 *      mlp_call() or used in loss expressions.  They appear as prev[]
 *      leaves of the graph.  The test owns and must free them — but
 *      only once, explicitly, NOT by the graph-walker.
 *
 *   3. TRANSIENT intermediate Values: the mul/add/relu/power/etc.
 *      nodes created during a forward pass or loss computation.
 *      They exist only for the duration of one epoch and must be freed
 *      after backward() has run.
 *
 * The graph-walker (free_graph_safe) must free category 3 and nothing
 * else.  It achieves this by refusing to descend into any node that
 * belongs to an "exclude" set supplied by the caller — which contains
 * both the MLP parameters AND the input/target Values.
 *
 * ORIGINAL BUG CHAIN
 * ──────────────────
 * 1. free_value() used a static visited[] array shared across calls.
 *    Resetting it at 10 000 entries discarded tracking state mid-walk,
 *    causing double-frees.
 *
 * 2. free_value() walked through prev[] into weight/bias Values and
 *    freed them.  On the next epoch mlp_zero_grad() wrote to freed
 *    memory → "Invalid write" in valgrind → segfault after epoch 1.
 *
 * 3. free_value(avg_loss) was followed by free_value(divisor) and
 *    free_value(losses[i]) — all of those nodes had already been freed
 *    by the first call → double-free cascade.
 *
 * 4. After free_graph_safe() was introduced it freed input/target
 *    leaf Values (they are prev[] children, not in the params set),
 *    and then the test called free(inputs[i][j]) again → double-free.
 */

/* ── free_graph_safe ─────────────────────────────────────────────────────── */
/*
 * Recursively frees every transient node reachable from `root`, EXCEPT for
 * any node present in `excl[]`.  Both MLP parameters and input/target Values
 * must be listed in excl[].
 */
static int in_set(Value* v, Value** set, int n) {
    for (int i = 0; i < n; i++) if (set[i] == v) return 1;
    return 0;
}

static void fgs_helper(Value* v,
                        Value** excl, int n_excl,
                        Value** visited, int* vs, int cap) {
    if (v == NULL) return;
    if (in_set(v, excl,    n_excl)) return; /* parameter or input leaf — skip */
    if (in_set(v, visited, *vs))    return; /* already freed in this walk */
    if (*vs >= cap) { fprintf(stderr, "fgs: visited overflow\n"); return; }
    visited[(*vs)++] = v;
    fgs_helper(v->prev[0], excl, n_excl, visited, vs, cap);
    fgs_helper(v->prev[1], excl, n_excl, visited, vs, cap);
    free(v);
}

/*
 * free_graph_safe(root, excl, n_excl)
 *
 * Free the transient computation graph rooted at `root`.
 * Nodes in excl[0..n_excl-1] are treated as persistent and are not freed.
 * The caller must separately free input/target Values that were excluded.
 */
static void free_graph_safe(Value* root, Value** excl, int n_excl) {
    if (root == NULL) return;
    const int cap = 200000;
    Value** visited = malloc(cap * sizeof(Value*));
    int vs = 0;
    fgs_helper(root, excl, n_excl, visited, &vs, cap);
    free(visited);
}

/*
 * build_excl_set: convenience helper — concatenates the params array and a
 * flat 2D inputs array into a single exclusion set.
 *
 * params[0..n_params-1]            : MLP weights and biases
 * inputs[n_inputs][n_features]     : input leaf Values
 * targets[n_targets]               : target leaf Values (may be NULL)
 */
static Value** build_excl(Value** params, int n_params,
                           Value* inputs[][3], int n_inputs,
                           Value** targets, int n_targets,
                           int* out_n) {
    int n = n_params + n_inputs * 3 + n_targets;
    Value** excl = malloc(n * sizeof(Value*));
    int k = 0;
    for (int i = 0; i < n_params;        i++) excl[k++] = params[i];
    for (int i = 0; i < n_inputs;  i++)
        for (int j = 0; j < 3; j++)      excl[k++] = inputs[i][j];
    for (int i = 0; i < n_targets;       i++) excl[k++] = targets[i];
    *out_n = k;
    return excl;
}

/* ── test_mlp_init ─────────────────────────────────────────────────────────── */
void test_mlp_init() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);

    int expected = (3*4 + 4) + (4*4 + 4) + (4*1 + 1);
    int actual   = mlp_n_params(&mlp);
    printf("MLP Init Test:\n");
    printf("  Expected params: %d\n", expected);
    printf("  Actual params:   %d (%s)\n\n", actual,
           actual == expected ? "PASS" : "FAIL");

    mlp_free(&mlp);
}

/* ── test_forward_pass ─────────────────────────────────────────────────────── */
void test_forward_pass() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);

    int     n_p    = mlp_n_params(&mlp);
    Value** params = mlp_parameters(&mlp);

    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0),  create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0),  create_value(1.0)},
        {create_value(1.0), create_value(1.0),  create_value(-1.0)}
    };

    /* Build exclusion set: params + inputs (no targets here) */
    int n_excl;
    Value** excl = build_excl(params, n_p, inputs, 4, NULL, 0, &n_excl);

    Value** outputs[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        printf("Forward Pass Test %d:\n", i+1);
        printf("  Expected output size: 1\n");
        printf("  Actual output size:   %d (%s)\n\n", mlp.layers[2].n_neurons,
               mlp.layers[2].n_neurons == 1 ? "PASS" : "FAIL");
    }

    for (int i = 0; i < 4; i++) {
        free_graph_safe(outputs[i][0], excl, n_excl); /* frees transient nodes only */
        free(outputs[i]);                              /* free the output array      */
    }
    /* Now free the excluded leaf Values that we own */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++) free(inputs[i][j]);

    free(excl);
    free(params);
    mlp_free(&mlp);
}

/* ── test_backward ─────────────────────────────────────────────────────────── */
void test_backward() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);

    int     n_p    = mlp_n_params(&mlp);
    Value** params = mlp_parameters(&mlp);

    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0),  create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0),  create_value(1.0)},
        {create_value(1.0), create_value(1.0),  create_value(-1.0)}
    };
    Value* targets[4] = {
        create_value(1.0), create_value(-1.0),
        create_value(-1.0), create_value(1.0)
    };

    /* Exclude params, inputs, AND targets from graph-walk frees */
    int n_excl;
    Value** excl = build_excl(params, n_p, inputs, 4, targets, 4, &n_excl);

    Value** outputs[4];
    Value*  losses[4];
    for (int i = 0; i < 4; i++) {
        outputs[i] = mlp_call(&mlp, inputs[i]);
        losses[i]  = power(sub(outputs[i][0], targets[i]), 2.0);
        free(outputs[i]); /* free the array; Values are in the loss graph */
    }

    mlp_zero_grad(&mlp);
    for (int i = 0; i < 4; i++) backward(losses[i]);

    int all_zero = 1;
    for (int i = 0; i < n_p; i++) {
        if (params[i]->grad != 0.0) { all_zero = 0; break; }
    }
    printf("Backward Pass Test:\n");
    printf("  All params have non-zero grad: %s (%s)\n\n",
           all_zero ? "NO" : "YES",
           all_zero ? "FAIL" : "PASS");

    for (int i = 0; i < 4; i++) {
        free_graph_safe(losses[i], excl, n_excl);
        for (int j = 0; j < 3; j++) free(inputs[i][j]);
    }
    for (int i = 0; i < 4; i++) free(targets[i]);

    free(excl);
    free(params);
    mlp_free(&mlp);
}

/* ── test_training ─────────────────────────────────────────────────────────── */
void test_training() {
    MLP mlp;
    int nouts[] = {4, 4, 1};
    mlp_init(&mlp, 3, nouts, 3);

    Value* inputs[4][3] = {
        {create_value(2.0), create_value(3.0),  create_value(-1.0)},
        {create_value(3.0), create_value(-1.0), create_value(0.5)},
        {create_value(0.5), create_value(1.0),  create_value(1.0)},
        {create_value(1.0), create_value(1.0),  create_value(-1.0)}
    };
    Value* targets[4] = {
        create_value(1.0), create_value(-1.0),
        create_value(-1.0), create_value(1.0)
    };

    int     n_params = mlp_n_params(&mlp);
    Value** params   = mlp_parameters(&mlp);

    /* Build the exclusion set once; it is reused every epoch */
    int n_excl;
    Value** excl = build_excl(params, n_params, inputs, 4, targets, 4, &n_excl);

    double* m = calloc(n_params, sizeof(double));
    double* v = calloc(n_params, sizeof(double));
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, lr = 0.02;

    for (int epoch = 0; epoch < 200; epoch++) {
        mlp_zero_grad(&mlp);

        Value* total_loss = create_value(0.0);

        for (int i = 0; i < 4; i++) {
            Value** output = mlp_call(&mlp, inputs[i]);
            Value*  diff   = sub(output[0], targets[i]);
            Value*  sq     = power(diff, 2.0);
            total_loss     = add(total_loss, sq);
            free(output);       /* only the pointer array; Values are in graph */
        }

        Value* divisor  = create_value(4.0);
        Value* avg_loss = truediv(total_loss, divisor);

        backward(avg_loss);

        for (int i = 0; i < n_params; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * params[i]->grad;
            v[i] = beta2 * v[i] + (1 - beta2) * pow(params[i]->grad, 2);
            double m_hat = m[i] / (1 - pow(beta1, epoch + 1));
            double v_hat = v[i] / (1 - pow(beta2, epoch + 1));
            params[i]->data -= lr * m_hat / (sqrt(v_hat) + eps);
        }

        printf("Training Step %d - Average Loss: %.8f\n", epoch + 1, avg_loss->data);

        /*
         * Free the epoch graph.  excl[] contains params + inputs + targets,
         * so the walker stops at all persistent nodes.  divisor and
         * total_loss are interior nodes reached automatically — do NOT
         * free them separately.
         */
        free_graph_safe(avg_loss, excl, n_excl);
    }

    /* Final evaluation */
    printf("\nFinal Training Results:\n");
    Value* final_total = create_value(0.0);
    for (int i = 0; i < 4; i++) {
        Value** output = mlp_call(&mlp, inputs[i]);
        Value*  diff   = sub(output[0], targets[i]);
        Value*  sq     = power(diff, 2.0);
        final_total    = add(final_total, sq);
        printf("Input %d: Predicted = %.8f, Actual = %.8f\n",
               i+1, output[0]->data, targets[i]->data);
        free(output);
    }
    Value* div4      = create_value(4.0);
    Value* final_avg = truediv(final_total, div4);
    printf("Final Average Loss: %.8f\n", final_avg->data);
    free_graph_safe(final_avg, excl, n_excl);

    free(m);
    free(v);
    free(excl);
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

