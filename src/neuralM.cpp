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


int main (int argc, char *argv[]){
	 try{

		 if (argc < 1) {
		        return 1;
		 }
		 int param = atoi(argv[1]);
		 clock_t begin = clock();

		 if(param==0){
			 NeuralSkeleton skeleton;
			 skeleton.learningRate=0.000003111111;
			 skeleton.momentum=0.79111;
			 skeleton.mCutoff=1250000;
			 skeleton.aCutoff=800000;
			 skeleton.sampleMax=12;
			 skeleton.sampleMin=2;
			 skeleton.init();
			 skeleton.generateTrainingData();

			 NeuralNetwork * nn = new NeuralNetwork(skeleton);
			 nn->createNetwork();
			 nn->iterate();
			 // nn->runAnnealingTrainingRound(skeleton.inputDataSize,249,15222222);
			 nn->pRun(false);

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

			 nn=0;
		 }else if(param==1){
			 NeuralSkeleton skeleton;

			 std::ifstream is("out.cereal", ios::in|ios::binary);
			 cereal::BinaryInputArchive iarchive(is);
			 iarchive(skeleton);

			 NeuralNetwork * nn = new NeuralNetwork(skeleton);
			 nn->createNetwork();
			 nn->setWeights();
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
