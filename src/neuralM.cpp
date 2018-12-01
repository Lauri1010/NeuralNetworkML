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

		 // ActivationFunctionTanh * af=new ActivationFunctionTanh();

		 clock_t begin = clock();
		 int neurons[]={3,20,20,20,20,1};
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

		 vector<vector<long double>>inputData;
		 inputData.push_back({0.101,0.16152,0.11196});
		 inputData.push_back({0.108,0.13089,0.13335});
		 int m=4000;
		 for(int u=2;u<m;u++){
			 long double ud=(long double)u;
			 long double d1=abs(sin(((inputData.at(u-1).at(0)+inputData.at(u-2).at(0)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.1, 0.005))+ud*0.01));
			 long double d2=abs(sin(((inputData.at(u-1).at(1)+inputData.at(u-2).at(1)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.07, 0.001))+ud*0.011));
			 long double d3=abs(sin(((inputData.at(u-1).at(2)+inputData.at(u-2).at(2)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.11, 0.007))+ud*0.0111));
			 inputData.push_back({d1,d2,d3});
		 }
		 int iSize=inputData.size();


		 vector<vector<long double>>idealData;

		 for(int t=0;t<iSize;t++){
			 // long double td=(long double)t;
			 long double ival=((inputData.at(t).at(0)+inputData.at(t).at(1)+inputData.at(t).at(2))/3);
/*			 long double ival2=abs(af->activationOutput(ival*0.0001));*/
			 idealData.push_back({
				 ival
			 });
		 }

/*
		 for (std::vector<int>::const_iterator i = neuralMap.begin(); i != neuralMap.end(); ++i)
		     std::cout << *i << ' ';

		 for(int i=0;i<inputData.size();i++){
			 cout << "input "<<inputData.at(i).at(0);
		 }
*/

		 double learningRate=0.000014;
		 double momentum=0.8;
		 int mCutoff=2000000;
		 double aCutoff=mCutoff-1000;
		 bool train=true;
		 double av=0.035;
/*		 long double bias=0.001;*/
		 // int ar=0;
		 // int sample=5;
		 int sampleMax=5;
		 int sampleMin=4;

		 if(param==0){
			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,av,mCutoff,aCutoff,sampleMax,sampleMin);
			 nn->createNetwork();
			 nn->iterate();
			 nn->pRun(false);
			 std::ofstream outfile("network.dat", std::ios::binary);
			 if (!outfile){
				       cout << "Unable to open for reading.\n";
				       return(1);
			  }
			  int ns=nn->neurons.size();
			  for(int prin=0;prin<ns;prin++){
/*				  char* bBytes = reinterpret_cast<char*>(&bias);*/
/*				  outfile.write(reinterpret_cast<const char*>(bBytes), sizeof(bBytes));*/
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
		 }else if(param==1){

			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,av,mCutoff,1,sampleMax,sampleMin);
			 nn->createNetwork();

			 ifstream inFile("network.dat", ios::in|ios::binary);
			 if (!inFile){
			       cerr << "File could not be opened." << endl;
			       exit(1);
			 }
			  int ns=nn->neurons.size();
			  for(int prin=0;prin<ns;prin++){
/*				  inFile.read(reinterpret_cast<char*>(&bias),sizeof(&bias));*/
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
			 nn->pRun(false);
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
