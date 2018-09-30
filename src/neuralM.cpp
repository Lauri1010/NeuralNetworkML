//============================================================================
// Name        : Neuralm.cpp
// Author      : Lauri Turunen
// Version     :
// Copyright   : Lauri Turunen
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
		 int neurons[]={3,9,9,9,42,1};
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
		 inputData.push_back({0.10000,0.11000,0.11000});
		 inputData.push_back({0.101,0.16152,0.11196});
		 inputData.push_back({0.108,0.13089,0.13335});
		 inputData.push_back({0.103,0.10134,0.10459});
		 inputData.push_back({0.109,0.1495,0.1552});
		 inputData.push_back({0.125,0.15299,0.1466});
		 inputData.push_back({0.136,0.18348,0.17438});
		 inputData.push_back({0.137,0.15366,0.17261});
		 inputData.push_back({0.138,0.14629,0.16507});
		 inputData.push_back({0.129,0.17531,0.14881});
		 inputData.push_back({0.135,0.10566,0.15649});
		 inputData.push_back({0.111,0.12559,0.09955});
		 inputData.push_back({0.112,0.11648,0.1295});
		 inputData.push_back({0.113,0.14161,0.15677});
		 inputData.push_back({0.119,0.15183,0.08472});


		 vector<vector<double>>idealData;
		 idealData.push_back({0.10667});
		 idealData.push_back({0.124826});
		 idealData.push_back({0.12408});
		 idealData.push_back({0.102976});
		 idealData.push_back({0.1379});
		 idealData.push_back({0.14153});
		 idealData.push_back({0.16462});
		 idealData.push_back({0.154423});
		 idealData.push_back({0.149786});
		 idealData.push_back({0.15104});
		 idealData.push_back({0.132383});
		 idealData.push_back({0.112046});
		 idealData.push_back({0.119326});
		 idealData.push_back({0.137126});
		 idealData.push_back({0.118516});

/*
		 for (std::vector<int>::const_iterator i = neuralMap.begin(); i != neuralMap.end(); ++i)
		     std::cout << *i << ' ';

		 for(int i=0;i<inputData.size();i++){
			 cout << "input "<<inputData.at(i).at(0);
		 }
*/

		 double learningRate=0.000001;
		 double momentum=0.0000001;
		 int mCutoff=25000000;
		 // int aCutoff=mCutoff/2;
		 bool train=true;
		 double av=0.06;

		 if(param==0){
			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,av,mCutoff);
			 nn->createNetwork();
			 double cHeat=1;
			 double cycles=1;
			 SimulatedAnnealing * sa = new SimulatedAnnealing(nn);
			 bool final=false;
			 for(int it=0;it < mCutoff;it++){
				    nn->iteration(final,it,false,false);
					if (nn->eIncreasing){
						nn->learningRate*=1.0001;
						nn->momentum=nn->momentum/2;
						if(nn->eIncreasingCount>100000){
							it=sa->runAnnealing(it,cHeat, cycles, nn->totalReturnValueP);
							cHeat=cHeat+0.05;
							cycles=cycles+0.05;
						}
					}else{
						nn->learningRate*=0.9999;
					}
					if(nn->totalReturnValueP < av && nn->totalReturnValueP > 0 ){
						break;
					}

			 }
			 nn->iteration(true,0,false,false);
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
				  char* bytes = reinterpret_cast<char*>(&nn->neurons.at(prin)->ao);
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
			  sa=0;

		 }else if(param==1){
			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,av,mCutoff);
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
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->ao),sizeof(&nn->neurons.at(prin)->ao));
				  int lSize=nn->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->fromNeuron), sizeof(nn->neurons.at(prin)->in.at(li)->fromNeuron));
					  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->weight), sizeof(&nn->neurons.at(prin)->in.at(li)->weight));
				  }

			  }

			 nn->resetNeurons();
			 nn->iteration(true,0,false,false);
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
