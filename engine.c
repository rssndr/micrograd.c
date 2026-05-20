// engine.c
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "engine.h"

void add_backward(Value* out) {
    Value *a = out->prev[0];
    Value *b = out->prev[1];
    a->grad += out->grad;
    b->grad += out->grad;
}

void mul_backward(Value* out) {
    Value *a = out->prev[0];
    Value *b = out->prev[1];
    a->grad += b->data * out->grad;
    b->grad += a->data * out->grad;
}

void power_backward(Value* out) {
    Value *a = out->prev[0];
    double b = atof(out->op + 1);
    a->grad += (b * pow(a->data, b - 1)) * out->grad;
}

void relu_backward(Value* out) {
    Value *a = out->prev[0];
    double leak = 0.01;
    a->grad += (out->data > 0 ? 1.0 : leak) * out->grad;
}

Value* create_value(double data) {
    Value* v = (Value*)malloc(sizeof(Value));
    if (v == NULL) return NULL;
    v->data = data;
    v->grad = 0.0;
    v->prev[0] = NULL;
    v->prev[1] = NULL;
    v->op[0] = '\0';
    v->backward = NULL;
    return v;
}

char* repr(Value* v) {
    static char vrepr[100];
    snprintf(vrepr, sizeof(vrepr), "Value(data=%f, grad=%f)", v->data, v->grad);
    return vrepr;
}

Value* add(Value* a, Value* b) {
    Value* out = create_value(a->data + b->data);
    if (out == NULL) return NULL;
    out->prev[0] = a;
    out->prev[1] = b;
    strcpy(out->op, "+");
    out->backward = add_backward;
    return out;
}

Value* mul(Value* a, Value* b) {
    Value* out = create_value(a->data * b->data);
    if (out == NULL) return NULL;
    out->prev[0] = a;
    out->prev[1] = b;
    strcpy(out->op, "*");
    out->backward = mul_backward;
    return out;
}

Value* power(Value* a, double b) {
    Value* out = create_value(pow(a->data, b));
    if (out == NULL) return NULL;
    out->prev[0] = a;
    out->prev[1] = NULL;
    snprintf(out->op, sizeof(out->op), "^%f", b);
    out->backward = power_backward;
    return out;
}

Value* relu(Value* a) {
    double leak = 0.01;
    Value* out = create_value(a->data < 0 ? leak * a->data : a->data);
    if (out == NULL) return NULL;
    out->prev[0] = a;
    out->prev[1] = NULL;
    strcpy(out->op, "ReLU");
    out->backward = relu_backward;
    return out;
}

Value* neg(Value* a) {
    return mul(a, create_value(-1));
}

Value* sub(Value* a, Value* b) {
    return add(a, neg(b));
}

Value* truediv(Value* a, Value* b) {
    return mul(a, power(b, -1));
}

/*
 * BUG (engine.c): build_topo() allocated a fixed 10,000-node DFS stack with
 * no overflow check.  A large epoch graph (hundreds of intermediate Values per
 * forward pass × 4 samples) can exceed this, silently writing past the end of
 * the heap allocation and corrupting memory.  backward() paired this with two
 * 1,000,000-element arrays (16 MB each) allocated on every call regardless of
 * actual graph size — wasteful and still potentially undersized for very deep
 * graphs.
 *
 * FIX: perform a lightweight first-pass count so all three working buffers are
 * allocated to exactly the size needed (with a small multiplier for DFS
 * headroom), and add an explicit overflow abort.
 */
static void count_nodes_helper(Value* v, Value** visited, int* sz, int cap) {
    Value** stk = malloc(cap * sizeof(Value*));
    int stk_sz = 0;
    stk[stk_sz++] = v;
    while (stk_sz > 0) {
        Value* node = stk[--stk_sz];
        if (node == NULL) continue;
        int seen = 0;
        for (int i = 0; i < *sz; i++) {
            if (visited[i] == node) { seen = 1; break; }
        }
        if (seen) continue;
        visited[(*sz)++] = node;
        for (int i = 0; i < 2; i++) {
            if (node->prev[i] != NULL) stk[stk_sz++] = node->prev[i];
        }
    }
    free(stk);
}

static int count_nodes(Value* v) {
    const int cap = 65536;
    Value** visited = malloc(cap * sizeof(Value*));
    int sz = 0;
    count_nodes_helper(v, visited, &sz, cap);
    free(visited);
    return sz;
}

void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size, int cap) {
    Value** stack = malloc(cap * sizeof(Value*));
    int stack_size = 0;
    stack[stack_size++] = v;

    while (stack_size > 0) {
        Value* node = stack[--stack_size];
        int is_visited = 0;
        for (int i = 0; i < *visited_size; i++) {
            if (visited[i] == node) { is_visited = 1; break; }
        }
        if (is_visited) continue;
        visited[(*visited_size)++] = node;
        for (int i = 1; i >= 0; i--) {
            if (node->prev[i] != NULL) {
                if (stack_size >= cap) {
                    fprintf(stderr, "build_topo: DFS stack overflow (cap=%d)\n", cap);
                    exit(EXIT_FAILURE);
                }
                stack[stack_size++] = node->prev[i];
            }
        }
    }
    for (int i = *visited_size - 1; i >= 0; i--) {
        topo[(*topo_size)++] = visited[i];
    }
    free(stack);
}

void backward(Value* v) {
    int n   = count_nodes(v);
    int cap = (n < 16 ? 16 : n) * 2;

    Value** topo    = malloc(cap * sizeof(Value*));
    Value** visited = malloc(cap * sizeof(Value*));
    int topo_size = 0, visited_size = 0;

    build_topo(v, topo, &topo_size, visited, &visited_size, cap);

    v->grad = 1.0;
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward != NULL) {
            topo[i]->backward(topo[i]);
        }
    }

    free(topo);
    free(visited);
}

