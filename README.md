# Multilayer Neural Network Beta
Low level Multilayer backpropagation neural network developed with c++, you can easily edit. For those you want a sufficiently advanced but compact and easy to edit program. 
Developed with high performance and high customizability in mind. It is very easy to customize different types of network structures as Neurons can be setup in any way you wish so long as you create the connection rules for the network (different network structures will be developed).
As such this is not a specially optimized network for the "standard" layered architecture, but can easily be edited to various topologies.


Features:
- Easy to edit multilayer network
- Portable skeleton struct enables easy sharing and loading from hard disk
- Unlimited input, hidden and output neurons
- Using two backpropagation learning systems (Gradient descent, Stochastic gradient descent)
- Simulated annealing (global optimal solution)
- Good performance (developed using c++, optimized) & highly customizable
- Saving and loading network from binary file
- Advanced testing training data generator (lots of hard peaks and valleys to learn)

![Neural network](https://www.ttaito.fi/images/trainedabit.png)

NOTE: I do not claim to be an expert in machine learning or mathematics. All you see here are things I've studied on my own or learned through experimentation.
Work is ongoing and this more than anything a labour of love. I simply enjoy developing this software, but want to share it here as it does work :)   

You must run the program on windows command prompt (win 10)
- NeuralM.exe 0 for training mode
- NeuralM.exe 1 for prediction mode

Note generating the training set may take several hours. The training set is saved to a binary file and then read for the prediction.

# Basic installation instructions: 
- use ide for instance: https://www.eclipse.org/downloads/packages/eclipse-ide-cc-developers/oxygen3a
- Install cygwin: https://www3.ntu.edu.sg/home/ehchua/programming/howto/EclipseCpp_HowTo.html
  MAKE sure you download all dependencies. 
  
  Compiled with windows Cygwin library. Note that this requires -fopenmp even though its not used
  
