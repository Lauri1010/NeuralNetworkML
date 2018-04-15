/*
 * neural.h
 *
 *  Created on: 25.2.2018
 *      Author: Lauri
 */

#ifndef NEURAL_H_
#define NEURAL_H_
#include <math.h>
#include <vector>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <algorithm>
#include <iomanip>
#include <float.h>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
class my_exception : public std::runtime_error {
    std::string msg;
public:
    my_exception(const std::string &arg, const char *file, int line) :
    std::runtime_error(arg) {
        std::ostringstream o;
        o << file << ":" << line << ": " << arg;
        msg = o.str();
    }
    ~my_exception() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }
};
#define throw_line(arg) throw my_exception(arg, __FILE__, __LINE__);
using namespace std;

double fRand(double fMax,double fMin)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

class ActivationFunction{
	double aOutput();
	double dFunction();

};

class ActicationFunctionTanh:ActivationFunction{
	public:
	ActicationFunctionTanh(){

	}

	double activationOutput(double d){
		return (exp(d*2.0)-1.0)/(exp(d*2.0)+1.0);
	}
	double dFunction(double d){
		return 1.0-pow(activationOutput(d), 2.0);
	}
};

class ActicationFunctionSigmoid:ActivationFunction{
	public:
	ActicationFunctionSigmoid(){

	}

	double aOutput(double d){
	       double e = exp(d);
	       e=e*9;
	       double y = - 1.0/(1.0+e);
	       return y;
	}
	double dFunction(double d){
		double e = exp(d);
		double y = - e/((1.0+e)*(1.0+e));
		return y;

	}
};


class Input{
	public:
	int fromNeuron;
	int fromLayer;
	int toLayer;
	double error;
	double errorSumIn;
	double delta;
	double weight;
	double pWeight;
	double inputValue;
	double wi;
	double wim;
	Input(int layer,int fromNeuron){
		this->fromLayer=layer-1;
		this->toLayer=layer;
		this->fromNeuron=fromNeuron;
		this->error=0.0;
		this->errorSumIn=0.0;
		this->delta=0.0;
		this->wi=0.52;
		this->wim=0.22;
		//this->weight = this->wi * fRand(this->wi*2.4,this->wi*2) - this->wim;
		// this->weight=0.4+fRand(0.15,0.1)+fRand(0.13,0.1)+fRand(0.12,0.1);
	    this->weight=fRand(0.5,0.4);
		this->pWeight= this->weight;
		this->inputValue=0.0;

	}
	void resetValues(){
		this->errorSumIn=0;
		this->error=0;
		this->inputValue=0;
	}
	double setInput(double input){
		this->inputValue=this->weight*input;
		return this->inputValue;
	}
	double setInputUw(double input){
		this->inputValue=input;
		return this->inputValue;
	}
	double getWeight(){
		return this->weight;
	}
	void sumError(double error){
		this->errorSumIn+=error;
	}
	void sumDelta(double delta){
		this->delta+=delta;
	}
	void setError(double error){
		this->error=error;
	}
	double calcWeightResult(){
		return this->inputValue*this->weight;
	}
	double getInput(){
		return this->inputValue;
	}
	void clearError(){
		this->error=0;
	}
	int getFrom(){
		return this->fromNeuron;
	}

	void adjustWeights(double learningRate,double momentum,bool ei){
		// w-1 * (previous weight).
		// pWeight= previous weight
		double a=this->weight;
		if(ei){
			learningRate*=7.5;
		}
		this->weight += (learningRate*this->errorSumIn);
		if(!ei){
			momentum*=10.52;
			this->weight +=(momentum*(this->weight-this->pWeight));
		}else{
			momentum*=0.051;
			this->weight +=(momentum*(this->weight-this->pWeight));
			this->weight +=fRand(0.00000015,0.00000001);
		}
		// this->weight = this->weight+((learningRate*this->errorSumIn)+(momentum*(this->weight-this->pWeight)))+(learningRate*0.05*this->delta);
		// this->weight = this->weight-((learningRate*this->error)+(momentum*(this->weight-this->pWeight)));
		// this->weight +=(learningRate*0.00001*delta);
		// this->weight +=error;
		// first round same as current weight;
		this->delta=0;
		this->errorSumIn=0;
		this->pWeight=a;
	}

};

class Neuron{
	 public:
	 double inputSum;
	 double outputSum;
	 int layer;
	 int id;
	 double wInput;
	 double delta;
	 int inputs;
	 double finalOutputError;
	 double activationOutput;
	 vector<unique_ptr<Input>> in;
	 unique_ptr<ActicationFunctionTanh> af;
	 double weightSum;
	 double currentError;
	 double errorSum;
	 double errorSumAbs;
	 double currentInputSum;
	 double currentWeightSum;

	 Neuron(int id,int layer,int inputs){
		 this->inputSum=0;
		 this->currentInputSum=0;
		 this->outputSum=0;
		 this->layer=layer;
		 this->inputs=inputs;
		 this->id=id;
		 this->wInput=0;
		 this->af=make_unique<ActicationFunctionTanh>();
		 this->delta=0;
		 this->activationOutput=0;
		 this->currentError=0;
		 this->errorSumAbs=0;
		 this->weightSum=0;
		 this->currentWeightSum=0;
		 this->errorSum=0;
		 this->finalOutputError=0;
	 };
	 int getId(){
		 return this->id;
	 }
	 double getOutput(){
		 this->activationOutput=this->af->activationOutput(this->currentInputSum);
		 return this->activationOutput;
	 }

	 double aOutput(){
		 if(this->layer>0){
			 this->activationOutput=this->af->activationOutput(this->currentInputSum);
		 }else if(this->layer==0){
			 this->activationOutput=this->in.at(0)->getInput();
		 }
		 return this->activationOutput;
	 }

	 void finalOutput(double expected,int iteration, int olocation,bool showOutput){
		 try{
			 this->activationOutput=this->currentInputSum;
			 double rError=sqrt(pow(expected-this->activationOutput,2));
			 this->currentError=rError;
			 this->errorSumAbs+=rError;

			 // cout << "Iteration: " << iteration << " Error at"<< olocation << " : "<< rError <<endl;
			 if(showOutput){
				 cout << "Iteration: " << iteration << " expected at neuron"<< olocation << ": " << expected << " output: "<< this->activationOutput <<endl;
			 }
		 }catch (const std::exception& ex) {
		 			 throw_line(ex.what());
		 			 exit(EXIT_FAILURE);
		 }
	 }

	 double getOutputStatic(){
		 return this->activationOutput;
	 }

	 void setOutputDelta(){
		 this->delta=this->currentError*af->dFunction(this->inputSum);
	 }

	 void setInputs(int inputIdStart,int inputIdEnd){
		 try{
			 if(this->layer>0){
				 int fromNeuron=inputIdStart;
				 for(int i=0;i<this->inputs;i++){
						this->in.push_back(make_unique<Input>(this->layer,fromNeuron));
						fromNeuron++;
				 }
			 }else if(this->layer==0){
				 this->in.push_back(make_unique<Input>(this->layer,-1));
			 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }

	 void setInput(int index,int fromCheck,double input){
		 try{
				 if(this->in.at(index)->fromNeuron==fromCheck){
					 if(this->layer==0){
						 this->in.at(index)->setInputUw(input);
					 }else{
						 this->currentInputSum+=this->in.at(index)->setInput(input);
						 this->currentWeightSum+=this->in.at(index)->weight;
					 }
				 }else{
					 throw_line("Improper input set");
					 exit(EXIT_FAILURE);
				 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }

	 void outputNeuronCalcError(int inputIndex,unique_ptr<Neuron> &pNeuron,double lRate,double momentum,bool eInc){
		 try{
			if(this->in.at(inputIndex)->fromNeuron==pNeuron->id){
				double delta=this->currentError*af->dFunction(this->currentInputSum);
				double error=delta*lRate*pNeuron.get()->activationOutput;
				this->in.at(inputIndex)->sumDelta(delta);
				this->in.at(inputIndex)->sumError(error);
			}else{
				throw new string("Improperly setup backwards propagation");
				exit(EXIT_FAILURE);
			}
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }
	 void hiddenNeuronCalcError(int inputIndex,int cnliIn,unique_ptr<Neuron> &pNeuron,unique_ptr<Neuron> &nNeuron,double lRate,double momentum,bool eInc){
		 try{
			 int pNeuronFromNeuron=this->in.at(inputIndex)->fromNeuron;
			 int nFromNeuron=nNeuron.get()->in.at(cnliIn)->fromNeuron;
			 if(pNeuronFromNeuron==pNeuron->id && nFromNeuron==this->id){
				 double delta=(nNeuron.get()->in.at(cnliIn)->delta*this->currentWeightSum*this->af->dFunction(this->currentInputSum))*-1;
				 double error=delta*lRate*pNeuron->getOutputStatic();
				 this->in.at(inputIndex)->sumDelta(delta);
				 this->in.at(inputIndex)->sumError(error);
			 }else{
				throw new string("Improperly setup backwards propagation");
			 	exit(EXIT_FAILURE);
			 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }


	 double getDelta(){
		 return this->delta;
	 }

	 void setNewIterationValues(){
		 this->currentError=0;
		 this->currentInputSum=0;
		 this->currentWeightSum=0;
		 this->inputSum=0;
		 this->weightSum=0;
		 this->errorSum=0;
/*	  	 for(unsigned int c=0;c<this->in.size();c++){
	    	this->in.at(c)->resetValues();
	     }*/
	 }


};


/**
 Feed forward: the output of each neuron in layer is passed to the input of the next layer neurons
 In backpropagation the error is calculated for each input
 In learning phase the weights are adjusted based on the error data (from output dissiminated to each layer)
 */
class NeuralNetwork{
	  public:
	  vector<vector<int>> neuronMap;
	  vector<vector<double>> inputData;
	  int inputDataLenght;
	  vector<vector <double>> idealData;
	  vector<vector<int>> layers;
	  int layerSize;
	  int idealDataLenght;
	  int neuronMapSize;
	  int neuronMapSizeM;
	  int inputDataSize;
	  int inputDataSizeM;
	  int it;
	  int cutoff;
	  double totalReturnValue;
	  double totalReturnValueP;
	  double mCutoff;
	  double totalOutput;
	  double errorPercentage;
	  double bestErrorRate;
	  double currentErrorValue;
	  double currentOutputSum;
	  double learningRate;
	  double oLearningRate;
	  double maxLearningRate;
	  double minLearningRate;
	  bool eIncreasing;
	  double momentum;
	  double oMomentum;
	  double minMomentum;
	  int eIncreaseCount;
	  bool train;
	  int dataLocation;
	  vector<unique_ptr<Neuron>> neurons;
	  int neuronsVSize;
	  int neuronsVSizeM;
	  double nDivider;
	  double cutoffErrorRate;
	  NeuralNetwork(vector<vector<int>> neuronMap,vector<vector<double>> inputData,vector<vector<double>> idealData,double learningRate,double momentum,bool train,int cutoff,double cErrorRate,double mCutoff){
		  cout.precision(10);
		  this->neuronMap=neuronMap;
		  this->neuronMapSize=neuronMap.size();
		  this->inputData=inputData;
		  this->inputDataLenght=inputData.size();
		  this->idealDataLenght=idealData.size();
		  this->idealData=idealData;
		  this->neuronMapSizeM=this->neuronMapSize-1;
		  this->learningRate=learningRate;
		  this->oLearningRate=learningRate;
		  this->maxLearningRate=learningRate*10;
		  this->minLearningRate=learningRate/10;
		  this->momentum=momentum;
		  this->oMomentum=momentum;
		  this->minMomentum=this->momentum/10000;
		  this->eIncreasing=false;
		  this->eIncreaseCount=0;
		  this->train=train;
		  this->cutoff=cutoff;
		  this->it=0;
		  this->totalReturnValue=1000;
		  this->totalReturnValueP=this->totalReturnValue;
		  this->mCutoff=mCutoff;
		  this->errorPercentage=0;
		  this->totalOutput=0;
		  this->bestErrorRate=0;
		  this->dataLocation=0;
		  this->nDivider=0;
		  this->currentErrorValue=0;
		  this->currentOutputSum=0;
		  this->inputDataSize=inputData.size();
		  this->inputDataSizeM=this->inputDataSize-1;
		  this->neuronsVSize=0;
		  this->neuronsVSizeM=0;
		  this->cutoffErrorRate=cErrorRate;
		  this->layerSize=0;
		  this->createNetwork();
	  }

	  void createNetwork(){
		  try{
		      if(this->neuronMapSize>0){
		    	 int id=0;
		    	 int sId=0;
		    	 int indx=0;
		    	 int layer=0;
				 while(layer<this->neuronMapSize){
					 	 	 for(int i=0;i<neuronMap.at(layer).at(0);i++){
						 	 	 this->neurons.push_back(make_unique<Neuron>(id,layer,neuronMap.at(layer).at(1)));
					 			 id++;
					 			 indx++;
					 	 	 }
					 	 	 layers.push_back({sId,id,indx});
					 	 	 indx=0;
					 	 	 sId=id;
					 	 	 layer++;
				 }
			     this->neuronsVSize=this->neurons.size();
			     this->neuronsVSizeM=this->neuronsVSize-1;
			     this->layerSize=this->layers.size();
			     for(auto &n : neurons){
			    	int lr=n->layer;
			    	if(lr>0){lr-=1;};
			    	n->setInputs(layers.at(lr).at(0), layers.at(lr).at(1));
			     }
				 cout << "Network created "<< endl;
		     }else{
		    	 throw new string("Neuron map improperly formed");
		    	 exit(EXIT_FAILURE);
		     }
		   }catch (const std::exception& ex) {
			     throw_line(ex.what());
				 exit(EXIT_FAILURE);
		   }

	  }


	  void iterate(){
		  if(this->train){
			  while(this->it < this->cutoff && this->totalReturnValue>this->cutoffErrorRate){
				  this->checkDataAndCleanUp();
				  this->iteration(false);
				  if(this->totalReturnValue==0){
					  break;
				  }
				  this->it++;
				  this->learningRate*=0.92;
				  this->momentum*=0.97;
			  }
			  this->totalReturnValue=0;
			  this->iteration(true);
			  this->calcFinalError();
		  }else{
			  this->iteration(true);
		  }
	  }


	  void iteration(bool predict){
		  try{
				 this->dataLocation=0;
				 while(this->dataLocation<this->inputDataSize){
					 this->feedForward(predict);
					 this->dataLocation++;
					 if(!predict){
						 this->backPropagate();
					 }
					 this->resetNeurons();
				 }
				 if(!predict){
					 this->learn();
				 }
		  }catch (const std::exception& ex) {
				 throw_line(ex.what());
				 exit(EXIT_FAILURE);
		  }

	  }


	  void setLearningRate(double lr){

	  }
	  void setMomentum(double sm){

	  }
	  void iLearningRate(double il){

	  }
	  void iMomentum(double im){

	  }
	  void dLearningRate(double dl){

	  }
	  void dMomentum(double dm){

	  }
	  void dLearningRateGradual(double limit){

	  }

	  void getNeuronErrorPercentage(Neuron n){

	  }

	  void feedForward(bool showOutput){
		 try{
			  int ifi=0;
			  int oDataLoc=0;
			  for(int c=0;c<this->neuronsVSize;c++){
					int layer=this->neurons.at(c)->layer;
					if(layer==0){
						this->neurons.at(c)->setInput(0,-1, this->inputData.at(this->dataLocation).at(ifi));
						ifi++;
					}else if(layer>0){
							int i=0;
							for(int ix=layers.at(layer-1).at(0);ix<layers.at(layer-1).at(1);ix++){
								this->neurons.at(c)->setInput(i,ix,this->neurons.at(ix)->aOutput());
								i++;
							}
							if(layer==this->neuronMapSizeM){
								this->neurons.at(c)->finalOutput(this->idealData.at(this->dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput);
								oDataLoc++;
							}
					}
			  }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	  }

	  void backPropagate(){
		  try{
			  int cNeuronIndex=this->neuronsVSizeM;
			  for(int layer=this->neuronMapSizeM;layer>0;layer--){
				  if(layer==this->neuronMapSizeM){
					  do{
						  // Neurons end at the previous layer
						  int previousNeuronId=this->layers.at(layer-1).at(1)-1;

						  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0 && previousNeuronId>-1;pi--,previousNeuronId--){
								  this->neurons.at(cNeuronIndex)->outputNeuronCalcError(pi, this->neurons.at(previousNeuronId), this->learningRate, this->momentum, this->eIncreasing);
						  };
						 // Move on to next neuron
						 cNeuronIndex--;
					  }while(this->neurons.at(cNeuronIndex)->layer==layer);
				  }else if(layer>0){

					  int nlin=this->layers.at(layer).at(2)-1;
					  do{

						  int nLayerCount=0;
						  // Neurons end at the previous layer
						  int previousNeuronId=this->layers.at(layer-1).at(1)-1;
						  int nextLayerNeuronId=this->layers.at(layer+1).at(1)-1;
						  int neuronsInNextLayer=this->layers.at(layer+1).at(2);

						  do{
							  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0 && previousNeuronId>-1;pi--,previousNeuronId--){
									  this->neurons.at(cNeuronIndex)->hiddenNeuronCalcError(pi,nlin,this->neurons.at(previousNeuronId),this->neurons.at(nextLayerNeuronId),this->learningRate,this->momentum,this->eIncreasing);
							  };
							  nextLayerNeuronId--;
							  nLayerCount++;
						  }while(nLayerCount<neuronsInNextLayer);

						 nlin--;
						 // Move on to next neuron
						 cNeuronIndex--;
					  }while(this->neurons.at(cNeuronIndex)->layer==layer);
				  }
			  }

		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }

	  }

	  void learn(){
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
				  for(int c=0;c<neurons.at(cNeuronIndex)->inputs;c++){
					  this->neurons.at(cNeuronIndex)->in.at(c)->adjustWeights(this->learningRate, this->momentum,this->eIncreasing);
				  }
			  	  if(layer==this->neuronMapSizeM){
					 // cout << "Iteration: "<< this->it << " current deviation from ideal results at location "<< outputLoc <<" : "<< this->neurons.at(cNeuronIndex)->errorSumAbs << endl;
					 this->totalReturnValue=this->totalReturnValue+this->neurons.at(cNeuronIndex)->errorSumAbs;
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void calcFinalError(){
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
			  	  if(layer==this->neuronMapSizeM){
					 // cout << "Iteration: "<< this->it << " current deviation from ideal results at location "<< outputLoc <<" : "<< this->neurons.at(cNeuronIndex)->errorSumAbs << endl;
					 this->totalReturnValue+=this->neurons.at(cNeuronIndex)->errorSumAbs;
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

		  cout << "Iteration: "<< this->it << " Final deviation from ideal results "<< this->totalReturnValue << endl;
	  }

	  void resetNeurons(){
		  int cNeuronIndex=this->neuronsVSizeM;
		  this->totalReturnValueP=0;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
			  	  this->neurons.at(cNeuronIndex)->setNewIterationValues();
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void checkDataAndCleanUp(){

/*		     if(this->totalReturnValue < this->totalReturnValueP && this->momentum < this->mCutoff){
		    	 this->momentum*=1.2;
		    	 this->eIncreaseCount=0;
		    	 this->eIncreasing=false;
		     }else if(this->totalReturnValue > this->totalReturnValueP && this->eIncreaseCount==0){
		    	 this->momentum=this->oMomentum*100;
		    	 this->eIncreaseCount++;
		    	 this->eIncreasing=true;
		     }else if(this->totalReturnValue > this->totalReturnValueP && this->eIncreaseCount>0){
		    	 this->momentum=this->oMomentum;
		    	 this->eIncreaseCount++;
		    	 this->eIncreasing=true;
		     }else{
		    	 this->eIncreasing=false;

		     }*/
		  	  if(this->it>0){
				  if(this->totalReturnValue < this->totalReturnValueP){
					this->eIncreasing=false;
				  }else if(this->totalReturnValue > this->totalReturnValueP){
					this->eIncreasing=true;
				  }else{
					this->eIncreasing=false;
				  }
	/*

				  for (vector<unique_ptr<Neuron>>::const_iterator i = this->neurons.begin(); i != this->neurons.end(); ++i){
					  i->get()->setNewIterationValues();
				  }
	*/
				 cout << "Iteration: "<< this->it << " current deviation from ideal results: "<< this->totalReturnValue << endl;
				 this->totalReturnValueP=this->totalReturnValue;
		  	 }
		     this->totalReturnValue=0;

	  }

};


#endif /* NEURAL_H_ */
