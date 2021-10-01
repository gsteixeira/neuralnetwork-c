SHELL := /bin/bash

default: neuralnet.c neuralnet.h
	gcc neuralnet.c -lm -O3 -o neuralnet

run:
	gcc neuralnet.c -lm -O3 -o neuralnet && ./neuralnet

clean:
	rm -f neuralnet
