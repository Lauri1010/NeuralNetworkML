# Multilayer Neural Network built with c++

## Features:
- Multithreaded for improved performance
- Header only: just include neural.h and you are ready to go in any project. 
- Easy to edit multilayer network for neural computing capable of running massive neural networks with excellent performance. Currently using 9600000 input nodes   (3 x 20 x 20 x 20 x 20 x 1). Note that at certain stage size does not matter. 
- Portable skeleton struct enables easy sharing and loading from hard disk
- Easy to load JSON data for predictions (upload data from database to JSON file in a cronjob for instance) using cereal library.  
- Unlimited input, hidden and output neurons
- Using two backpropagation learning systems (Gradient descent, Stochastic gradient descent)
- Simulated annealing (global optimal solution)
- Very high performance single thread. Able to go through 315000 training iterations (stochastic) in one minute
- Saving and loading network from binary file
- Training data generator for demonstration purposes (lots of hard peaks and valleys to learn)

## Running the Neural Network
Using the program is really easy and straightforward. Training mode example:  

```c++
// Create metadata of the network (with default values)
NeuralSkeleton skeleton;
skeleton.init();
skeleton.generateTrainingData();

// Create the network
unique_ptr<NeuralNetwork> nn = make_unique<NeuralNetwork>(skeleton);
nn->createNetwork();
nn->iterate();
nn->pRun();

// Store the network as metadata to a binary file using cereal
int ns=nn->neurons.size();
for(int prin=0;prin<ns;prin++){
		int lSize=nn->neurons.at(prin)->in.size();
		for(int li=0;li<lSize;li++){
				skeleton.setInputWeight(nn->neurons.at(prin)->in.at(li)->weight);
		}
}
skeleton.validateNetwork();
std::ofstream os("out.cereal", std::ios::binary);
cereal::BinaryOutputArchive oarchive(os);
oarchive(skeleton);

```
**Note that** you need to compile the code before running the program. The .exe files in this repo may not work in your particular computer (build in amd processor in windows 10 and optimized for native). Also you shoud start with modest settings.

**An example**
![neural network result](https://raw.githubusercontent.com/Lauri1010/NeuralNetworkML/master/trained.png)

For the network to fully learn the data several hundreds of thousands of iterations may be needed. But as can be seen above the network learns a rather challenging dataset with lots of peaks and valleys and some randomness remarkably well with with 180k training iterations. 


## The principle in neural network predictions: 
First load training data to JSON (from a database for instance) in the provided format. Then use the saved binary of the network after training to predict with new similar data (for instance stock index historic and future data). 
Note that tuning for different datasets may take some time as you need to adjust the learning rate and other parameters. To improve training you need to have plenty of training data. 

### Building and running in windows
In windows you need to install gycwin. 
- Use IDE. For instance: https://www.eclipse.org/downloads/packages/eclipse-ide-cc-developers/oxygen3a
- Install cygwin: https://www3.ntu.edu.sg/home/ehchua/programming/howto/EclipseCpp_HowTo.html
  MAKE sure you download all dependencies.
  
You must run the program on windows command prompt (win 10)
NeuralM.exe 0 for training mode
NeuralM.exe 1 for prediction mode

Note generating the training set may take several hours. The training set is saved to a binary file and then read for the prediction. Also be carefull not to set the learning rate too high in training (you will see what happens). 
  
### You can also build your own compiler in windows with the latest gcc:
**1. Run basic cygwin setup (installs the cygwin tools)**

Using the setup.exe file (64 or 32). 

**2. Install dependencies**

setup-x86_64.exe --disable-buggy-antivirus -q -P wget -P gcc-g++ -P make -P diffutils -P libmpfr-devel -P libgmp-devel -P libmpc-devel

**3. Download the latest gcc**

You can use home folder for instance.

wget http://www.nic.funet.fi/pub/gnu/ftp.gnu.org/pub/gnu/gcc/gcc-9.1.0/gcc-9.1.0.tar.gz 

tar xf gcc-9.1.0.tar.gz

**4. Configure the build**

mkdir build-gcc

cd build-gcc

../gcc-9.1.0/configure --program-suffix=-9.1.0 --enable-languages=c,c++ --disable-bootstrap --disable-shared

**5. Build the compiler**

make

make install

### Building and running the neural network in linux:
**You can build the sources for instance with:**

g++ -fopenmp -O3 -std=c++14 -Ofast -march=native -mtune=native -ftree-vectorize -ffast-math -frename-registers -floop-nest-optimize -Wall -I/path-to-includes/includes -c neuralm.cpp -o neuralm

This worked just fine in google cloud virtual server. Your system may have differences in the setup (and with c compilation).  

**running the program in linux**
for training
Run ./neuralm 0 
for prediction
./neuralm 1 


