//============================================================================
// Name        : Neuralm.cpp
// Author      : Lauri Turunen
// Version     :
// Copyright   : Lauri Turunen
// Description :
//============================================================================

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
#include <algorithm>
#include <omp.h>

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
	public:
	ActivationFunction(){};
	virtual ~ActivationFunction(){delete this;};
	virtual double activationOutput(double d) = 0;
	virtual double dFunction(double d) = 0;
};

class ActivationFunctionLinear final: public ActivationFunction{
	public:
	ActivationFunctionLinear(){};
	~ActivationFunctionLinear(){delete this;};
	double activationOutput(double d){
		return d;
	}
	double dFunction(double d){
		return 1;
	}
};


class ActivationFunctionTanh final: public ActivationFunction{
	public:
	ActivationFunctionTanh(){};
	~ActivationFunctionTanh(){delete this;};
	double activationOutput(double d){
		return (exp(d*2.0)-1.0)/(exp(d*2.0)+1.0);
	}
	double dFunction(double d){
		return 1.0-pow(activationOutput(d), 2.0);
	}
};

class ActivationFunctionSigmoid final: public ActivationFunction{
	public:
	ActivationFunctionSigmoid(){};
	~ActivationFunctionSigmoid(){delete this;};
	double activationOutput(double d){
		   double e = exp(d)*9;
	       return 1.0/(1.0+e);
	}
	double dFunction(double d){
		double e = exp(d);
		return e/((1.0+e)*(1.0+e));
	}
};

class ActivationFunctionRectifiedRelu final: public ActivationFunction{
	public:
	ActivationFunctionRectifiedRelu(){};
	~ActivationFunctionRectifiedRelu(){delete this;};
	double activationOutput(double d){
		d=max(d,0.0);
		return d;
	}
	double dFunction(double d){
		if(d<0){d=0;}else if(d>=0){d=1;};
		return d;
	}
};

class ActivationFunctionSoftPlus final: public ActivationFunction{
	public:
	ActivationFunctionSoftPlus(){};
	~ActivationFunctionSoftPlus(){delete this;};
	double activationOutput(double d){
		return log(1+pow(d,2));
	}
	double dFunction(double d){
		return 1/(1+pow(d,2));
	}
};


class Input{
	public:
	int fromNeuron;
	int fromLayer;
	int toLayer;
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
		this->errorSumIn=0.0;
		this->delta=0.0;
		this->wi=0.52;
		this->wim=0.22;
		// this->weight = this->wi * fRand(this->wi*2.4,this->wi*2) - this->wim;
		// this->weight=0.4+fRand(0.15,0.1)+fRand(0.13,0.1)+fRand(0.12,0.1);
	    // this->weight=fRand(0.1,0.04)+fRand(0.007,0.004)+0.01;
		// this->weight=0.05+fRand(0.045,0.038);
		// this->weight=fRand(0.65,0.33)+0.2;
		// this->weight=fRand(0.35,0.3)+fRand(0.15,0.09)+fRand(0.05,0.02)+0.015;
		this->weight=0.05;
		// this->weight=0.5+fRand(0.45,0.01);
		this->pWeight= this->weight;
		this->inputValue=0.0;

	}
	void resetValues(){
		this->errorSumIn=0;
		this->inputValue=0;
	}
	double setInput(double input){
		double iv=this->weight*input;
		this->inputValue=iv;
		return iv;
	}
	void setInputUw(double input){
		this->inputValue=input;
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
	double calcWeightResult(){
		return this->inputValue*this->weight;
	}
	double getInput(){
		return this->inputValue;
	}
	int getFrom(){
		return this->fromNeuron;
	}

	void adjustWeights(double learningRate,double momentum,bool ei){
		double a=this->weight;

		if(ei){
			learningRate*=10.5;
			momentum*=0.01;
		}
		this->weight += learningRate*this->errorSumIn+(this->delta*momentum);

		this->delta=0;
		this->errorSumIn=0;
		this->pWeight=a;
	}

	/*void adjustWeights(double learningRate,double momentum,bool ei){
			double a=this->weight;
			// double di=0.001;
			this->weight += learningRate*this->errorSumIn;
			this->delta=0;
			this->errorSumIn=0;
			this->pWeight=a;
	}*/

};

class Neuron{
	 public:
	 double inputSum;
	 double outputSum;
	 int layer;
	 int id;
	 bool func;
	 double wInput;
	 double delta;
	 int inputs;
	 double finalOutputError;
	 double ao;
	 vector<unique_ptr<Input>> in;
	 unique_ptr<ActivationFunction> af;
	 double weightSum;
	 double errorSumAbs;
	 double currentInputSum;
	 double currentWeightSum;

	 Neuron(int id,int layer,int inputs,int func,bool pl){
		 this->inputSum=0;
		 this->currentInputSum=0;
		 this->outputSum=0;
		 this->layer=layer;
		 this->inputs=inputs;
		 this->func=func;
		 this->id=id;
		 this->wInput=0;
		 this->delta=0;
		 this->ao=0;
		 this->errorSumAbs=0;
		 this->weightSum=0;
		 this->currentWeightSum=0;
		 this->finalOutputError=0;
		 if(func==0){
			 this->af=make_unique<ActivationFunctionSoftPlus>();
		 }else if(func==1){
			 if(layer==1){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else if(layer==2){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else if(pl){
				 this->af=make_unique<ActivationFunctionTanh>();
			 }else{
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }
			 /*
			 if(pl){
				 this->af=make_unique<ActivationFunctionRectifiedRelu>();
			 }else if(layer % 2 == 0){
				 this->af=make_unique<ActivationFunctionTanh>();
			 }else{
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }*/
		 }else if(func==2){
			 this->af=make_unique<ActivationFunctionLinear>();
		 }
	 };
	 void setErrorSumAbs(double es){
		 this->errorSumAbs=es;
	 }

	 int getId(){
		 return this->id;
	 }

	 void aOutput(){
		 this->ao=this->af->activationOutput(this->currentInputSum);
	 }
     // this->neurons.at(c)->finalOutput(this->idealData.at(dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput,this->totalReturnValue);
	 void finalOutput(double expected,int iteration, int olocation,bool showOutput,double& eSum){
		 try{
			 // this->setErrorSubAbs(expected-this->activationOutput);
			 double es=sqrt(pow(expected-this->ao,2));
			 this->setErrorSumAbs(es);
			 eSum+=es;
			 // cout << "Iteration: " << iteration << " Error at"<< olocation << " : "<< rError <<endl;
			 if(showOutput){
				 cout << "Iteration: " << iteration << " expected at neuron"<< olocation << ": " << expected << " output: "<< this->ao <<endl;
			 }
		 }catch (const std::exception& ex) {
		 			 throw_line(ex.what());
		 			 exit(EXIT_FAILURE);
		 }
	 }

	 double getOutputStatic(){
		 return this->ao;
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
						 this->currentInputSum+=input;
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
				double delta=this->errorSumAbs*af->dFunction(this->currentInputSum);
				double error=delta*lRate*pNeuron.get()->ao;
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
				 // double delta=nNeuron.get()->in.at(cnliIn)->delta*this->currentWeightSum*this->af->dFunction(this->currentInputSum);
				 double error=delta*lRate*pNeuron->getOutputStatic();
				 this->in.at(inputIndex)->sumDelta(delta);
				 this->in.at(inputIndex)->sumError(error);
			 }else{
				 throw_line("Inputs don't match");
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
		 this->currentInputSum=0;
		 this->currentWeightSum=0;
		 this->inputSum=0;
		 this->weightSum=0;
		 /*
	  	 for(unsigned int c=0;c<this->in.size();c++){
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
	  double totalReturnValue;
	  double totalReturnValueP;
	  int mCutoff;
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
	  int eIncreasingCount;
	  double momentum;
	  double oMomentum;
	  double minMomentum;
	  int eIncreaseCount;
	  bool train;
	  vector<unique_ptr<Neuron>> neurons;
	  int neuronsVSize;
	  int neuronsVSizeM;
	  double nDivider;
	  double cutoffErrorRate;
	  NeuralNetwork(vector<vector<int>> neuronMap,vector<vector<double>> inputData,vector<vector<double>> idealData,double learningRate,double momentum,bool train,double cErrorRate,int mcf){
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
		  this->mCutoff=mcf;
		  this->minMomentum=this->momentum/10000;
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
		  this->eIncreaseCount=0;
		  this->train=train;
		  this->it=0;
		  this->totalReturnValue=0;
		  this->totalReturnValueP=this->totalReturnValue;
		  this->errorPercentage=0;
		  this->totalOutput=0;
		  this->bestErrorRate=0;
		  this->nDivider=0;
		  this->currentErrorValue=DBL_MAX;
		  this->currentOutputSum=0;
		  this->inputDataSize=inputData.size();
		  this->inputDataSizeM=this->inputDataSize-1;
		  this->neuronsVSize=0;
		  this->neuronsVSizeM=0;
		  this->cutoffErrorRate=cErrorRate;
		  this->layerSize=0;
	  }

	  void resetNetworkForLoaded(){
		  cout.precision(10);
		  this->inputDataLenght=inputData.size();
		  this->idealDataLenght=idealData.size();
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
		  this->eIncreaseCount=0;
		  this->train=false;
		  this->it=0;
		  this->totalReturnValue=0;
		  this->totalReturnValueP=this->totalReturnValue;
		  this->errorPercentage=0;
		  this->totalOutput=0;
		  this->bestErrorRate=0;
		  this->nDivider=0;
		  this->currentErrorValue=0;
		  this->currentOutputSum=0;
		  this->inputDataSize=inputData.size();
		  this->inputDataSizeM=this->inputDataSize-1;
		  this->neuronsVSize=0;
		  this->neuronsVSizeM=0;
		  this->cutoffErrorRate=1;
		  this->layerSize=0;
	  }

	  void createNetwork(){
		  try{
		      if(this->neuronMapSize>0){
		    	 int id=0;
		    	 int sId=0;
		    	 int indx=0;
		    	 int layer=0;
		    	 int func=0;
		    	 bool pl=false;
				 while(layer<this->neuronMapSize){
					 	     if(layer==1){func++;};
					 	     if(layer==this->neuronMapSizeM){func++;};
					 	 	 for(int i=0;i<neuronMap.at(layer).at(0);i++){
						 	 	 this->neurons.push_back(make_unique<Neuron>(id,layer,neuronMap.at(layer).at(1),func,pl));
					 			 id++;
					 			 indx++;
					 	 	 }
					 	 	 layers.push_back({sId,id,indx});
					 	 	 indx=0;
					 	 	 sId=id;
					 	 	 layer++;
					 	 	 if(layer==this->neuronMapSizeM-1){
					 	 		pl=true;
					 	 	 }
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
		    	 throw_line("Improperly constructed neural network");
		    	 exit(EXIT_FAILURE);
		     }
		   }catch (const std::exception& ex) {
			     throw_line(ex.what());
				 exit(EXIT_FAILURE);
		   }

	  }

	  /*
	  void iterate(){
		  if(this->train){

			  while(this->it < this->cutoff && this->totalReturnValue>this->cutoffErrorRate){
					  this->checkDataAndCleanUp();
					  this->iteration(false);
					  if(this->totalReturnValue==0 && this->it > 0){
						  break;
					  }
					  this->it++;
			  }

			  this->totalReturnValue=0;

		  }else{
			  this->iteration(true);
		  }
	  }*/

	  void iteration(bool predict,int iteration,bool mt, bool suppress){
		  try{
			  	 this->it=iteration;

			  	 if(mt){
			  		 /*
					 int dataLocation=0;
					 int mxth=omp_get_max_threads();
					 while(dataLocation<this->inputDataSize){
						 #pragma omp parallel
						 {
							if(dataLocation<this->inputDataSize){
									int tn=omp_get_thread_num();
									if(tn < mxth){
										// #pragma omp single
										#pragma omp critical
										this->feedForward(predict,dataLocation);
									}
								#pragma omp atomic
								dataLocation++;
								// cout << "Iteration: "<< dataLocation;
							}else{
								#pragma omp barrier
							}
						 }
					 }*/
			  		/*
			  		omp_set_dynamic(0);
			  		omp_set_num_threads(2);
					#pragma omp parallel
					{
						#pragma omp for schedule(static)
			  			for(int dataLocation=0;dataLocation<this->inputDataSize;dataLocation++){
			  						  			this->feedForward(predict,dataLocation);
			  									this->backPropagate();
			  									this->learn();
			  									this->resetNeurons();
			  			}
			  			this->checkDataAndCleanUp(suppress);
					}*/

			  	 }else{
			  		for(int dataLocation=0;dataLocation<this->inputDataSize;dataLocation++){
			  			this->feedForward(predict,dataLocation);
			  			if(!predict){
			  				this->backPropagate();
			  			}
			  			this->resetNeurons();
			  		}
			  		if(!predict){
			  			this->learn();
			  		}
			  		this->checkDataAndCleanUp(suppress);

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

	  void feedForward(bool showOutput,int dataLocation){
		 try{
			 if(dataLocation<this->inputDataLenght){
				  int ifi=0;
				  int oDataLoc=0;

					 for(int c=0;c<this->neuronsVSize;c++){
							int layer=this->neurons.at(c)->layer;
							if(layer==0){
								this->neurons.at(c)->setInput(0,-1, this->inputData.at(dataLocation).at(ifi));
								ifi++;
							}else if(layer>0){
									int i=0;
									for(int ix=layers.at(layer-1).at(0);ix<layers.at(layer-1).at(1);ix++){
										this->neurons.at(c)->setInput(i,ix,this->neurons.at(ix)->getOutputStatic());
										i++;
									}

							}
							this->neurons.at(c)->aOutput();
							if(layer==this->neuronMapSizeM){
								this->neurons.at(c)->finalOutput(this->idealData.at(dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput,this->totalReturnValue);
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
						  // The next neuron input index used to link to upper input id of the fromNeuron
						  int nlin=this->layers.at(layer).at(2)-1;
						  do{

							  int nLayerCount=0;
							  // Neurons end at the previous layer
							  int nextLayerNeuronId=this->layers.at(layer+1).at(1)-1;
							  int neuronsInNextLayer=this->layers.at(layer+1).at(2);

							  do{
								  int previousNeuronId=this->layers.at(layer-1).at(1)-1;
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

	  /***
	   * After feed forward and backpropagated will set the weights.
	   *
	   */
	  void learn(){
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
				  for(int c=0;c<neurons.at(cNeuronIndex)->inputs;c++){
					  this->neurons.at(cNeuronIndex)->in.at(c)->adjustWeights(this->learningRate, this->momentum,this->eIncreasing);
				  }
			  	  if(layer==this->neuronMapSizeM){
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void calcFinalError(){
		  /*
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  while(layer==this->neuronMapSizeM){
			  	  if(layer==this->neuronMapSizeM){
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }else{
			  		  break;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  };*/

		  cout << "Iteration: "<< this->it << " Final deviation from ideal results "<< this->totalReturnValue << endl;
	  }

	  /***
	   * Will reset the current data location errors
	   *
	   */
	  void resetNeurons(){
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
			  	  this->neurons.at(cNeuronIndex)->setNewIterationValues();
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void checkDataAndCleanUp(bool suppress){
			this->learningRate=this->learningRate/(1+(this->it/this->mCutoff));

		    if(this->totalReturnValueP>0){
					  if((this->totalReturnValue < this->totalReturnValueP)){
						this->eIncreasing=false;
						this->eIncreasingCount=0;
					  }else if((this->totalReturnValue > this->totalReturnValueP)){
						this->eIncreasing=true;
						// this->learningRate*=1.005;
					  }else if((this->totalReturnValue == this->totalReturnValueP)){
						this->eIncreasing=true;
						// this->learningRate*=1.005;
					  }
		  	 }
			 if(this->eIncreasing){
				this->eIncreasingCount++;
			 }

			 this->totalReturnValueP=this->totalReturnValue;

		     if(!suppress){
					 cout << "Iteration: "<< this->it << " current deviation from ideal results: "<< this->totalReturnValue << endl;
			 }
		     this->totalReturnValue=0;

	  }

/*	  void simulatedAnnealing(double heat,double cycles){
		   double currentTemp=heat;
		   double stoptemp=heat*0.65;
		   double pError;
		   double peError;
	       double bpError;
		   double ratio = exp(log(stoptemp / heat) / (cycles - 1));



		  vector<unique_ptr<Neuron>> currentBestSet;
		  copy(this->neurons.begin(),this->neurons.end(),back_inserter(currentBestSet));

	  }*/




};

class SimulatedAnnealing{
	public:
	NeuralNetwork * cnb;


	SimulatedAnnealing(NeuralNetwork * cn){
		this->cnb=cn;
	};

	int runAnnealing(int i,double heat,double cycles,double oError){
		double currentTemp=heat;
		double stoptemp=heat*0.25;
		double pError=oError;
		double peError=oError;
		double ratio = exp(log(stoptemp / heat) / (cycles - 1));
		bool stop=false;
		double stopRate=DBL_MAX;
		double stopRatet=0.5;

		do{
			int c=0;
			do{
				this->randomize(currentTemp,heat);
				int ti=0;
				do{
					peError=pError;
					this->cnb->iteration(false, i, false,true);
					pError=this->cnb->totalReturnValueP;
					cout << "Annealing iteration: "<< i << " Deviation from ideal results: "<< pError << endl;
					i++;
					stopRate=pError/oError;
					if(stopRate<stopRatet || pError <= this->cnb->cutoffErrorRate){
						stop=true;
					}
					ti++;
				}while(pError < peError && !stop);
				c++;
			}while(c < cycles && !stop);

			currentTemp *= ratio;

		}while(currentTemp>stoptemp && !stop);

		return i;
	}

	void randomize(double currentTemp,double startTemp){
		int i=0;
		int ns=cnb->neurons.size();
		while(i<ns){
			if(cnb->neurons.at(i)->layer>1){
				int si=cnb->neurons.at(i)->in.size();
				for(int c=0;c<si;c++){
					double add = 0.000001 - fRand(0.0000003,0.0000001);
					add /= startTemp;
					add *= currentTemp;
					cnb->neurons.at(i)->in.at(c)->weight+=add;
				}
			}
			i++;
		}

	}

};

#endif /* NEURAL_H_ */
