//============================================================================
// Name        : Neuralm.cpp
// Author      : Lauri Turunen
// Version     :
// Copyright   : 
// Description :
//============================================================================


#include "neural.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iterator>
#include <fstream>

template<typename T>
std::ostream& binary_write(std::ostream& stream, const T& value){
    return stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
};

template<typename T>
std::istream & binary_read(std::istream& stream, T& value){
    return stream.read(reinterpret_cast<char*>(&value), sizeof(T));
}


int main (int argc, char *argv[]){
	 try{

		 if (argc < 1) {
		        return 1;
		 }
		 int param = atoi(argv[1]);

		 clock_t begin = clock();
		 int neurons[]={3,6,3,3,3,3,3,3,3,3,2,2,2};
		 // int neurons[]={2,3,3,3,1};
		 vector<vector<int>>neuralMap;
		 int nSize=sizeof(neurons)/sizeof(neurons[0]);
		 for(int a = 0; a < nSize; a++){
			 if(a>0){
				 neuralMap.push_back({neurons[a],neurons[a-1]});
			 }else if(a==0){
				 neuralMap.push_back({neurons[a],1});
			 }

		 }

		 vector<vector<double>>inputData;
		 inputData.push_back({0.10000,0.10000,0.20000});
		 inputData.push_back({0.10100,0.16152,0.11196});
		 inputData.push_back({0.10200,0.23089,0.13335});
		 inputData.push_back({0.10300,0.30134,0.10459});
		 inputData.push_back({0.10400,0.14950,0.09552});
		 inputData.push_back({0.10500,0.17299,0.10660});
		 inputData.push_back({0.10600,0.20348,0.17438});
		 inputData.push_back({0.10700,0.25366,0.17261});
		 inputData.push_back({00.10800,0.14629,0.17507});
		 inputData.push_back({0.10900,0.20531,0.14881});
		 inputData.push_back({0.11000,0.10566,0.15649});
		 inputData.push_back({0.11100,0.16559,0.07955});
		 inputData.push_back({0.11200,0.19648,0.09950});
		 inputData.push_back({0.11300,0.14161,0.15677});
		 inputData.push_back({0.11400,0.18183,0.08472});

		 vector<vector<double>>idealData;
		 idealData.push_back({0.13333,0.079661});
		 idealData.push_back({0.17853,0.077907});
		 idealData.push_back({0.18302,0.110801});
		 idealData.push_back({0.17905,0.021499});
		 idealData.push_back({0.17416,0.142693});
		 idealData.push_back({0.15896,0.005919});
		 idealData.push_back({0.16249,0.121739});
		 idealData.push_back({0.14994,0.078183});
		 idealData.push_back({0.12673,0.072126});
		 idealData.push_back({0.10858,0.005775});
		 idealData.push_back({0.11607,0.046935});
		 idealData.push_back({0.15602,0.097248});
		 idealData.push_back({0.17093,0.071620});
		 idealData.push_back({00.15683,0.100342});
		 idealData.push_back({0.17537,0.141281});

/*
		 for (std::vector<int>::const_iterator i = neuralMap.begin(); i != neuralMap.end(); ++i)
		     std::cout << *i << ' ';

		 for(int i=0;i<inputData.size();i++){
			 cout << "input "<<inputData.at(i).at(0);
		 }
*/

		 double learningRate=0.0001;
		 double momentum=0.001;
		 double mCutoff=58000000;
		 bool train=true;
		 int cutoff=mCutoff-10;
		 int av=0.35;

		 if(param==0){

			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,cutoff,av,mCutoff);
			 nn->createNetwork();

			 for(int it=0;it<cutoff;it++){
					nn->checkDataAndCleanUp();
					nn->iteration(false,it,false);
			 }

			 nn->calcFinalError();
			 nn->iteration(true,cutoff,false);
			 std::ofstream outfile("network.dat", std::ios::binary);
			 if (!outfile){
				       cout << "Unable to open for reading.\n";
				       return(1);
			  }
			  int ns=nn->neurons.size();
			  for(int prin=0;prin<ns;prin++){
				  outfile.write(reinterpret_cast<char*>(&nn->neurons.at(prin)->id), sizeof(int));
				  outfile.write(reinterpret_cast<char*>(&nn->neurons.at(prin)->inputs), sizeof(int));
				  outfile.write(reinterpret_cast<char*>(&nn->neurons.at(prin)->layer), sizeof(int));
				  outfile.write(reinterpret_cast<char*>(&nn->neurons.at(prin)->func), sizeof(int));
				  char* bytes = reinterpret_cast<char*>(&nn->neurons.at(prin)->activationOutput);
				  outfile.write(reinterpret_cast<char*>(bytes), sizeof(bytes));
				  int lSize=nn->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  outfile.write(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->fromNeuron), sizeof(int));
					  char* bytes2 = reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->weight);
					  outfile.write(reinterpret_cast<const char*>(bytes2), sizeof(bytes2));
				  }
			  }
			  outfile.close();

			  nn=0;

		 }else if(param==1){
			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,cutoff,av,mCutoff);
			 nn->createNetwork();

			 ifstream inFile("network.dat", ios::in|ios::binary);
			 if (!inFile){
			       cerr << "File could not be opened." << endl;
			       exit(1);
			 }
			  int ns=nn->neurons.size();
			  for(int prin=0;prin<ns;prin++){

				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->id), sizeof(int));
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->inputs), sizeof(int));
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->layer), sizeof(int));
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->func), sizeof(int));
				  // char* bytes = reinterpret_cast<char*>(&nn->neurons.at(prin)->activationOutput);
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->activationOutput),sizeof(&nn->neurons.at(prin)->activationOutput));
				  int lSize=nn->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->fromNeuron), sizeof(nn->neurons.at(prin)->in.at(li)->fromNeuron));
					  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->weight), sizeof(&nn->neurons.at(prin)->in.at(li)->weight));
				  }

			  }

			 nn->resetNeurons();
			 nn->iteration(true,0,false);
			 nn=0;
		 }
		 clock_t end = clock();
		 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		 cout << "Elapsed time in seconds: "<< elapsed_secs;
		 return 0;

	 }catch (const std::runtime_error &ex) {
		         cout << ex.what() << std::endl;
	 }

}
