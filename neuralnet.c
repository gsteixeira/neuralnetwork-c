// C implementation of a simple Feedforward Neural Network
//
// Author: Gustavo Selbach Teixeira
//
// gcc neuralnet.c -lm -O3
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
// The neural network object's definitions are on neuralnet.h.
#include "neuralnet.h"

// Main function
int main() {
    srand (time(NULL));
    // input data
    double *predicted;
    double inputs[4][2] = {{0.0, 0.0},
                           {1.0, 0.0},
                           {0.0, 1.0},
                           {1.0, 1.0}};
    double outputs[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};
    int input_size = sizeof(inputs[0])/sizeof(double);
    int output_size = sizeof(outputs[0])/sizeof(double);

    NeuralNetwork nn;
    nn = NewNeuralNetwork(input_size, output_size, 4);
    train(nn, input_size, output_size, inputs, outputs, 10000);
    
    for (int i=0; i<4; i++) {
        predicted = predict(nn, inputs[i]);
        printf("input: [%f, %f] predicted: %f output: %f\n",
                    inputs[i][0], inputs[i][1], predicted[0],
                    nn.output_layer.values[0]);
    }
}

// Layer constructor method
Layer NewLayer (int size, int parent_size) {
    int i, j;
    Layer layer;
    layer.values = malloc(size * sizeof(double));
    layer.bias = malloc(size * sizeof(double));
    layer.deltas = malloc(size * sizeof(double));
    layer.weights = malloc(parent_size * sizeof(double *));
    layer.size = size;
    layer.connection_size = parent_size;
    
    for (i=0; i<size; i++) {
        layer.values[i] = get_random();
        layer.bias[i] = get_random();
    }
    for (i=0; i<parent_size; i++) {
        layer.weights[i] = malloc(size * sizeof(double));
        for (j=0; j<size;j++) {
            layer.weights[i][j] = get_random();
        }
    }
    return layer;
}

// The NeuralNetwork constructor method
NeuralNetwork NewNeuralNetwork (int input_size,
                       int output_size,
                       int hidden_size) {
    NeuralNetwork nn;
    nn.input_layer = NewLayer(input_size, 1);
    nn.hidden_layer = NewLayer(hidden_size, input_size);
    nn.output_layer = NewLayer(output_size, hidden_size);
    
    nn.input_size = input_size;
    nn.hidden_size = hidden_size;
    nn.output_size = output_size;
    
    nn.learning_rate = 0.1;
    return nn;
}

// The logistical sigmoid function
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

// The derivative of sigmoid function
double d_sigmoid(double x) {
    return x * (1 - x);
}

// Feed inputs to forward through the network
void set_inputs (NeuralNetwork nn, double inputs[]) {
    for (int i=0; i<nn.input_size; i++) {
        nn.input_layer.values[i] = inputs[i];
    }
}

// The activation function
void activation_function(Layer source, Layer target) {
    double activation;
    for (int j=0; j<target.size; j++) {
        activation = target.bias[j];
        for (int i=0; i<source.size; i++) {
            activation += (source.values[i] * target.weights[i][j]);
        }
        target.values[j] = sigmoid(activation);
    }
}

// Compute the delta for the output layer
void calc_delta_output(NeuralNetwork nn, double expected[]) {
    double errors;
    for (int i=0; i<nn.output_layer.size; i++) {
        errors = (expected[i] - nn.output_layer.values[i]);
        nn.output_layer.deltas[i] = (errors 
                            * d_sigmoid(nn.output_layer.values[i]));
    }
}

// Compute the deltas between layers
void calc_deltas(Layer source, Layer target) {
    double errors;
    for (int j=0; j<target.size; j++) {
        errors = 0.0;
        for (int k=0; k<source.size; k++) {
            errors += (source.deltas[k] * source.weights[j][k]); // ERROR j, k
        }
        target.deltas[j] = (errors * d_sigmoid(target.values[j]));
    }
}

// Update the weights
void update_weights(Layer source, Layer target, double learning_rate) {
    for (int j=0; j<source.size; j++) {
        source.bias[j] += (source.deltas[j] * learning_rate);
        for (int k=0; k<target.size; k++) {
            source.weights[k][j] += (target.values[k] * source.deltas[j] * learning_rate);
        }
    }
}

// Neural network main loop
void train(NeuralNetwork nn,
           size_t input_size, size_t output_size,
           double inputs[][input_size],
           double outputs[][output_size],
           int n_epochs) {
    int i, e;
    int num_training_sets = 4;
    
    for (e=0; e<n_epochs; e++) {
        for (i=0; i<num_training_sets; i++) {
            set_inputs(nn, inputs[i]);
            // Forward pass
            activation_function(nn.input_layer, nn.hidden_layer);
            activation_function(nn.hidden_layer, nn.output_layer);
            // Show results
            printf("%d Input: [%f, %f] Expected: [%f] Output: %f\n",
                    e, inputs[i][0], inputs[i][1], outputs[i][0],
                    nn.output_layer.values[0]);
            // Back propagation
            // calculate the deltas
            calc_delta_output(nn, outputs[i]);
            calc_deltas(nn.output_layer, nn.hidden_layer);
            // from output to hidden layer
            update_weights(nn.output_layer, nn.hidden_layer,
                           nn.learning_rate);
            update_weights(nn.hidden_layer, nn.input_layer,
                           nn.learning_rate);
        }
    }
}

// Make a prediction. To be used once the network has been trained
double *predict (NeuralNetwork nn, double inputs[]) {
    set_inputs(nn, inputs);
    activation_function(nn.input_layer, nn.hidden_layer);
    activation_function(nn.hidden_layer, nn.output_layer);
    return nn.output_layer.values;
}

// returns a double between 0 and 1
double get_random(){
    return ((double)rand()/(double)(RAND_MAX)) * 1;
}
