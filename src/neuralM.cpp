//============================================================================
// Name        : NeuralM.cpp
// Author      : Lauri
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================


#include "neural.h"
using namespace std;

 int main (int argc, char *argv[])
 {
	 try{
		 int neurons[]={4,10,7,5,7,2};
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
		 inputData.push_back({0.1,0.1,0.2,0.5});
		 inputData.push_back({0.1,0.15,0.1,0.5});
		 inputData.push_back({0.11,0.18,0.21,0.5});
		 inputData.push_back({0.09,0.07,0.05,0.01});
		 inputData.push_back({0.17,0.08,0.05,0.04});
		 inputData.push_back({0.05,0.01,0.2,0.01});
		 inputData.push_back({0.15,0.12,0.01,0.01});
		 inputData.push_back({0.17,0.13,0.2,0.55});
		 inputData.push_back({0.07,0.03,0.07,0.05});
		 inputData.push_back({0.09,0.08,0.02,0.03});
		 inputData.push_back({0.11,0.07,0.15,0.7});
		 inputData.push_back({0.13,0.08,0.07,0.05});
		 inputData.push_back({0.17,0.1,0.09,0.095});
		 inputData.push_back({0.10,0.12,0.04,0.02});
		 inputData.push_back({0.05,0.07,0.04,0.08});

		 vector<vector<double>>idealData;
		 idealData.push_back({0.2,0.1});
		 idealData.push_back({0.22,0.11});
		 idealData.push_back({0.27,0.15});
		 idealData.push_back({0.11,0.10});
		 idealData.push_back({0.14,0.07});
		 idealData.push_back({0.05,0.05});
		 idealData.push_back({0.17,0.08});
		 idealData.push_back({0.18,0.055});
		 idealData.push_back({0.05,0.07});
		 idealData.push_back({0.12,0.09});
		 idealData.push_back({0.14,0.11});
		 idealData.push_back({0.07,0.1});
		 idealData.push_back({0.1,0.09});
		 idealData.push_back({0.11,0.08});
		 idealData.push_back({0.07,0.075});

/*
		 for (std::vector<int>::const_iterator i = neuralMap.begin(); i != neuralMap.end(); ++i)
		     std::cout << *i << ' ';

		 for(int i=0;i<inputData.size();i++){
			 cout << "input "<<inputData.at(i).at(0);
		 }
*/


		 double learningRate=0.0019;
		 double momentum=0.0012;
		 double maxMomentum=0.55;
		 bool train=true;
		 int cutoff=5000000;

		 NeuralNetwork * nn=new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,cutoff,0.11,maxMomentum);
		 nn->iterate();
		 nn=0;
		 return 0;

	 }catch (const std::runtime_error &ex) {
		         cout << ex.what() << std::endl;
     }

 }
