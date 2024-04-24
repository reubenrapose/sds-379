# Demonstrating Neural Networks as Universal Approximators
### Completed for Certificate in Scientific Computation and Data Sciences
#### Author: Reuben Rapose

This project was done in the course SDS 379R to complete the certificate in Scientific Computation and Data Sciences at UT Austin. The goal was to perform a research project in Scientific Computation and showcase results. The research was done on neural networks, and their ability to approximate functions and solutions to differential equations. This repository holds the code that was used.

#### Some notes about the individual files: 
In general, the code needs a fair amount of editing to perform specific functions (approximating a specific function over a specific domain). The existing code is mostly set up for convenience to recreate the process of training models for examples included in the paper, and nothing else.

The file `python/torch_nn.py` runs assuming there is a directory called `outputs/` visible from the working directory. This is theoretically where outputs from training would go (outputs being a progress tracker on the loss function). The file `python/keras_nn.py` is not run-ready. The file is simply included to provide evidence of attempts to use TensorFlow, however the author deemed it inferior to PyTorch in terms of custom loss functions for derivatives. Eventually, keras was abandoned in favor of the package torch.nn.

As for the C++ files, to train a model:
1. In `cpp/nn_functions.cpp`, the global variables `nodes_per_layer` and `layers` define the size of the network.
2. In `cpp/main.cpp` lies the pair of functions `trainGradientDescent` and `trainGradientDescentPDE`. Inside there is a single call to a loss function, and a gradient descent function. This needs to be edited.
3. In `cpp/main.cpp`, the domain of the function or differential equation is defined in the `main` function. This would also need to be edited.
4. Lastly, in `cpp/main.cpp`, the `main` function holds the potential function calls one might use to train or predict a model, as well as headers of files for storing outputs. This can be edited to fit appropriate needs. 

Please contact the author at reubenrapose@utexas.edu if there are any questions or issues with accessing the repository.
