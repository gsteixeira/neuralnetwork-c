#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

// Prototipes
// The Layer object
typedef struct Layers {
    double *values;
    double *bias;
    double *deltas;
    double **weights;
    int size;
    int connection_size;
} Layer;

// The NeuralNetwork object
typedef struct NeuralNetworks {
    Layer input_layer;
    Layer hidden_layer;
    Layer output_layer;
    double learning_rate;
    int input_size;
    int hidden_size;
    int output_size;
} NeuralNetwork;

// Prototipes
Layer NewLayer (int, int);
NeuralNetwork NewNeuralNetwork(int, int, int);
double get_random ();
double sigmoid (double);
double d_sigmoid (double);
double *predict (NeuralNetwork, double[]);
void set_inputs (NeuralNetwork, double[]);
void activation_function (Layer, Layer);
void calc_delta_output (NeuralNetwork, double[]);
void calc_deltas (Layer, Layer);
void update_weights (Layer, Layer, double);
void train (NeuralNetwork, size_t x, size_t y, double[][x], double[][y],int);

#endif
