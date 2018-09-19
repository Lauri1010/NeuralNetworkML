# Multilayer Neural Network
Multilayered backpropagation neural network developed with c++

You must run the program on windows command prompt (win 10)
- NeuralM.exe 0 for training mode
- NeuralM.exe 1 for prediction mode

Note generating the training set may take several hours. The training set is saved to a binary file and then read for the prediction.

# Basic installation instructions: 
- use ide for instance: https://www.eclipse.org/downloads/packages/eclipse-ide-cc-developers/oxygen3a
- Install cygwin: https://www3.ntu.edu.sg/home/ehchua/programming/howto/EclipseCpp_HowTo.html
  MAKE sure you download all dependencies. 
  
  Compiled with windows Cygwin libraries 
  
  
# Compiling the code example 

g++ -O3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"src/neuralm.d" -MT"src/neuralm.o" -o "src/neuralm.o" "../src/neuralm.cpp"

g++ -fopenmp -o "NeuralM.exe"  ./src/neuralm.o   
