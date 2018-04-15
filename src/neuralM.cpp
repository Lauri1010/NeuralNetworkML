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
		 int neurons[]={2,7,7,1};
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
		 inputData.push_back({0.1,0.1});
		 inputData.push_back({0.1,0.15});
		 inputData.push_back({0.11,0.18});
		 inputData.push_back({0.09,0.07});
		 inputData.push_back({0.17,0.08});
		 inputData.push_back({0.05,0.01});
		 inputData.push_back({0.15,0.12});
		 inputData.push_back({0.17,0.13});
		 inputData.push_back({0.07,0.03});
		 inputData.push_back({0.09,0.08});
		 inputData.push_back({0.11,0.07});
		 inputData.push_back({0.13,0.08});
		 inputData.push_back({0.17,0.1});
		 inputData.push_back({0.10,0.12});
		 inputData.push_back({0.05,0.07});

		 vector<vector<double>>idealData;
		 idealData.push_back({0.2});
		 idealData.push_back({0.22});
		 idealData.push_back({0.27});
		 idealData.push_back({0.11});
		 idealData.push_back({0.14});
		 idealData.push_back({0.05});
		 idealData.push_back({0.17});
		 idealData.push_back({0.18});
		 idealData.push_back({0.05});
		 idealData.push_back({0.12});
		 idealData.push_back({0.14});
		 idealData.push_back({0.07});
		 idealData.push_back({0.1});
		 idealData.push_back({0.11});
		 idealData.push_back({0.07});

/*
		 for (std::vector<int>::const_iterator i = neuralMap.begin(); i != neuralMap.end(); ++i)
		     std::cout << *i << ' ';

		 for(int i=0;i<inputData.size();i++){
			 cout << "input "<<inputData.at(i).at(0);
		 }
*/


		 double learningRate=0.057;
		 double momentum=0.052;
		 double maxMomentum=0.55;
		 bool train=true;
		 int cutoff=50000000;

		 NeuralNetwork * nn=new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,cutoff,0.51,maxMomentum);
		 nn->iterate();
		 nn=0;
		 return 0;

	 }catch (const std::runtime_error &ex) {
		         cout << ex.what() << std::endl;
     }

 }
