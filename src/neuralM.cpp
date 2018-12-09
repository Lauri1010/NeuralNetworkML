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
		 int neurons[]={3,20,20,20,20,1};
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
			 long double ival=((inputData.at(t).at(0)+inputData.at(t).at(1)+inputData.at(t).at(2))/3);
			 idealData.push_back({
				 ival
			 });
		 }

		 if(param==0){
			 double learningRate=0.00000000011111;
			 double momentum=0.79;
			 int mCutoff=15000;
			 double aCutoff=25000;
			 bool train=true;
			 double av=0.015;
			 int sampleMax=125;
			 int sampleMin=2;

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
				  char* bytes0 = reinterpret_cast<char*>(&nn->neurons.at(prin)->ao);
				  outfile.write(bytes0, sizeof(bytes0));
				  int lSize=nn->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  char* bytes1 = reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->weight);
					  outfile.write(bytes1, sizeof(bytes1));
				  }
			  }
			  outfile.close();
			  nn=0;
		 }else if(param==1){
			 double learningRate=0.00000000011111;
			 double momentum=0.79;
			 int mCutoff=2;
			 bool train=true;
			 double av=0.015;
			 int sampleMax=125;
			 int sampleMin=2;

			 NeuralNetwork * nn = new NeuralNetwork(neuralMap,inputData,idealData,learningRate,momentum,train,av,mCutoff,1,sampleMax,sampleMin);
			 nn->createNetwork();
			 ifstream inFile("network.dat", ios::in|ios::binary);
			 if (!inFile){
			       cerr << "File could not be opened." << endl;
			       exit(1);
			 }
			  int ns=nn->neurons.size();
			  for(int prin=0;prin<ns;prin++){
				  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->ao), sizeof(&nn->neurons.at(prin)->ao));
				  int lSize=nn->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  inFile.read(reinterpret_cast<char*>(&nn->neurons.at(prin)->in.at(li)->weight), sizeof(double));
				  }

			  }
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
