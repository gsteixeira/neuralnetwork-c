# A Neural Network in Go

Simple feed forward neural network in C

## usage:

If you just wanna try it:
```shell
    make run
    # or
    make
    ./neuralnet
```

## Create a neural network:

Create a network telling the size (nodes) of earch layer.
```c
    NeuralNetwork nn;
    nn = NewNeuralNetwork(input_size, output_size, hidden_layer_size);
    // Train it
    train(nn, input_size, output_size, inputs, outputs, 10000);
    // Now, make predictions
    double foo[2] = {0, 1};
    predicted = predict(nn, foo);
    printf("predicted: %f \n", predicted);
    // predicted: ~ 0.99;
```


