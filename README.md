# Neural computing: Deep Learning Multilayer Neural Network
Developed and optimized with c++
NOTE Optimized and compiled with: -O3 -Ofast -march=native -mtune=native -ftree-vectorize -ffast-math -frename-registers -fopenmp -std=c++17

## Features:
- Easy to edit multilayer network for neural computing
- Portable skeleton struct enables easy sharing and loading from hard disk
- Unlimited input, hidden and output neurons
- Using two backpropagation learning systems (Gradient descent, Stochastic gradient descent)
- Simulated annealing (global optimal solution)
- Very high performance single thread. Able to go through 50 000 training iterations in one minute
- Saving and loading network from binary file
- Advanced testing training data generator (lots of hard peaks and valleys to learn)

![Neural network](https://www.ttaito.fi/images/trainedabit.png)

NOTE: I do not claim to be an expert in machine learning or mathematics, but all techniques used in this program are constructed using tried and tested published AI algorithms: Gradient descent, Stochastic gradient descent, Simulated annealing. 

You must run the program on windows command prompt (win 10)
- NeuralM.exe 0 for training mode
- NeuralM.exe 1 for prediction mode

Note generating the training set may take several hours. The training set is saved to a binary file and then read for the prediction.

# Basic installation instructions: 
- use ide for instance: https://www.eclipse.org/downloads/packages/eclipse-ide-cc-developers/oxygen3a
- Install cygwin: https://www3.ntu.edu.sg/home/ehchua/programming/howto/EclipseCpp_HowTo.html
  MAKE sure you download all dependencies. 
  
  Compiled with windows Cygwin library. Note that this requires -fopenmp even though its not used
  
