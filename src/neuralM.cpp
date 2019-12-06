//============================================================================
// Name        : Neuralm.cpp
// Author      : Lauri Turunen
// Version     :
// Copyright   : Lauri Turunen
// Description :
//============================================================================


#include "neural.h"

int main (int argc, char *argv[]){
	 try{
		 if (argc < 1) {
		        return 1;
		 }
		 int param = atoi(argv[1]);
		 clock_t begin = clock();

		 if(param==0){
			 try{
				 NeuralSkeleton skeleton;
				 skeleton.learningRate=0.0000001551111;
				 skeleton.momentum=0.001545188;
				 skeleton.mCutoff=400000;
				 skeleton.m=2500;
				 skeleton.aCutoff=72000;
				 skeleton.sampleMax=100;
				 skeleton.sampleMin=20;
				 skeleton.init();
				 // Needs to be set even if it is overwritten
				 skeleton.generateTrainingData();

				 std::ofstream jFile("data.json",ios::out);
				 cereal::JSONOutputArchive jArchive(jFile);
				 jArchive(CEREAL_NVP(skeleton.inputData),CEREAL_NVP(skeleton.idealData));
				 jFile.close();
				 /*
				 std::ofstream jFile2("ideal.json",ios::out);
				 cereal::JSONOutputArchive jArchive2(jFile2);
				 jArchive2(CEREAL_NVP(skeleton.idealData));
				 jFile2.close();

				 exit(0); */

				 // Note } at end of file. You may need to add it to the file
				 /*
				 std::ifstream dataJson("data.json");
				 cereal::JSONInputArchive jsonInputAr(dataJson);
				 jsonInputAr(CEREAL_NVP(skeleton.inputData),CEREAL_NVP(skeleton.idealData));
				 dataJson.close();


				 for(unsigned int i=0;i<skeleton.inputData.size();i++){
					 printf("location: %i input : %.16g  \n",i,skeleton.inputData.at(i).at(0));
				 }

				 for(unsigned int i=0;i<skeleton.idealData.size();i++){
					 printf("location: %i ideal : %.16g  \n",i,skeleton.idealData.at(i).at(0));
				 }
				 exit(0);
				 */

				 unique_ptr<NeuralNetwork> nn = make_unique<NeuralNetwork>(skeleton);
				 nn->createNetwork();
				 nn->iterate();
				 nn->pRun();

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
			 }catch (int e)
			 {
			     cout << "An exception occurred. Exception Nr. " << e << '\n';
			 }
		 }else if(param==1){
			 try{

			 NeuralSkeleton skeleton;

			 std::ifstream is("out.cereal", ios::in|ios::binary);
			 cereal::BinaryInputArchive iarchive(is);
			 iarchive(skeleton);
			 skeleton.setInputDataMax();

			 // Note that for prediction ideal data can be zero (it is only used for training)
			 // You can load input data in JSON format and predict with the network. You can for instance load data from database into JSON format and use it here
			 /*

			 std::ifstream inputJson("json-file-name.json");
			 cereal::JSONInputArchive jsonInputAr(inputJson);
			 jsonInputAr(CEREAL_NVP(skeleton.inputData));
			 inputJson.close();

			 */

			 unique_ptr<NeuralNetwork> nn = make_unique<NeuralNetwork>(skeleton);
			 nn->createNetwork();
			 nn->setWeights();
			 nn->pRun();
			 }catch (int e)
			 {
			     cout << "An exception occurred. Exception Nr. " << e << '\n';
			 }
		 }
		 clock_t end = clock();
		 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		 cout << "Elapsed time in seconds: "<< elapsed_secs;
		 return 0;

	 }catch (const std::runtime_error &ex) {
		         cout << ex.what() << std::endl;
	 }

}
