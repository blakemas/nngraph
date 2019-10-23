# NNGraph

This repository contains an implementation of algorithms in "Learning Nearest Neighbor Graphs from Noisy
Distance Samples" by Mason et al., 2019. The data_gen file contains methods to produce the Euclidean embeddings 
tested in the simulations in the paper, and the anntri.py contains a class implementing all algorithms in the
paper. The algorithm takes in a general distance matrix which it uses for error computation and to pass to 
the oracle to gather samples from. All code is in python and assumes Python 3. The implementation requires numpy, 
scipy, sklearn, and matplotlib. 

For a simple demo, please view Example_experiment.ipynb in this repo
