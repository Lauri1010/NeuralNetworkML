# Neural computing: Multilayer Neural Network with simulated annealing - beta
Low level Multilayer backpropagation neural network developed and optimized with c++. Note that the development branch may be signifigantly more advanced, untill all of the changes are pulled to the master (will update this branch once in two months as a release). 

## Features:
- Easy to edit multilayer network for neural computing
- Portable skeleton struct enables easy sharing and loading from hard disk
- Unlimited input, hidden and output neurons
- Using two backpropagation learning systems (Gradient descent, Stochastic gradient descent)
- Simulated annealing (global optimal solution)
- **UPDATE: Improved performance: capable of training one million rows of 4000 item training set in 35 minutes (500k in 17). Tested with slowest ryzen 1400, gycwin compiler with: g++ -O3 -Ofast -ftree-vectorize -ffast-math -frename-registers**
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
  
