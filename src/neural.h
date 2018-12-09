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

long double fRand(long double fMax,long double fMin)
{
    long double f = (long double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

class ActivationFunction{
	public:
	ActivationFunction(){};
	virtual ~ActivationFunction(){delete this;};
	virtual long double activationOutput(long double d) = 0;
	virtual long double dFunction(long double d) = 0;
};

class ActivationFunctionLinear final: public ActivationFunction{
	public:
	ActivationFunctionLinear(){};
	~ActivationFunctionLinear(){delete this;};
	long double activationOutput(long double d){
		return d;
	}
	long double dFunction(long double d){
		return 1;
	}
};


class ActivationFunctionTanh final: public ActivationFunction{
	public:
	ActivationFunctionTanh(){};
	~ActivationFunctionTanh(){delete this;};
	long double activationOutput(long double d){
		return (exp(d*2.0)-1.0)/(exp(d*2.0)+1.0);
	}
	long double dFunction(long double d){
		return 1.0-pow(activationOutput(d), 2.0);
	}
};

class ActivationFunctionSigmoid final: public ActivationFunction{
	public:
	ActivationFunctionSigmoid(){};
	~ActivationFunctionSigmoid(){delete this;};
	long double activationOutput(long double d){
		   long double e = exp(d)*9;
	       return - 1.0/(1.0+e);
	}
	long double dFunction(long double d){
		long double e = exp(d);
		return - e/((1.0+e)*(1.0+e));
	}
};

class ActivationFunctionRectifiedRelu final: public ActivationFunction{
	public:
	ActivationFunctionRectifiedRelu(){};
	~ActivationFunctionRectifiedRelu(){delete this;};
	long double activationOutput(long double d){
		if(d<0){d=0;}
		return d;
	}
	long double dFunction(long double d){
		if(d<0){d=0;}else if(d>=0){d=1;};
		return d;
	}
};

class ActivationFunctionSoftPlus final: public ActivationFunction{
	public:
	ActivationFunctionSoftPlus(){};
	~ActivationFunctionSoftPlus(){delete this;};
	long double activationOutput(long double d){
		return log(1+pow(d,2));
	}
	long double dFunction(long double d){
		return 1/(1+pow(d,2));
	}
};

class ActivationFunctionSin final: public ActivationFunction{
	public:
	ActivationFunctionSin(){};
	~ActivationFunctionSin(){delete this;};
	long double activationOutput(long double d){
		return sin(d);
	}
	long double dFunction(long double d){
		return cos(d);
	}
};

class ActivationFunctionSinc final: public ActivationFunction{
	public:
	ActivationFunctionSinc(){};
	~ActivationFunctionSinc(){delete this;};
	long double activationOutput(long double d){
		if(d==0){
			return 1;
		}else{
			return (sin(d)/d);
		};
	}
	long double dFunction(long double d){
		if(d==0){
			return (sin(d)/d);
		}else{
			return (cos(d)/d)-(sin(d)/d);
		}

	}
};


class Input{
	public:
	int fromNeuron;
	int fromLayer;
	int toLayer;
	long double errorSumIn;
	long double delta;
	long double weight;
	long double pWeight;
	long double inputValue;
	long double wi;
	long double wim;
	long double bias;
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
		if(layer>0){
			this->weight=0.01+fRand(0.0065,0.006388)+fRand(0.0035,0.003288);
		}else{
			this->weight=1;
		}
		// this->weight=fRand(0.0065,0.006488)+0.001;
		// this->weight=fRand(0.0015,0.001)+fRand(0.015,0.0135)+fRand(0.0005,0.004)+0.00015;
		// this->weight=fRand(0.05,0.04995);
		// this->weight=0.01;
		this->pWeight= this->weight;
		this->inputValue=0.0;
		this->bias=0.01;
	}
	void resetValues(){
		this->errorSumIn=0;
		this->inputValue=0;
	}
	void setInput(long double input){
		this->inputValue=this->weight*input;
	}

	void setInputUw(long double input){
/*		this->weight=input/(((input-1)/input)*-1)*(1+input);*/
		this->inputValue=this->weight*input;
	}
	long double getWeight(){
		return this->weight;
	}
	void sumError(long double error){
		this->errorSumIn+=error;
	}

	void sumBias(long double bv){
			this->bias+=bv;
	}
	void sumDelta(long double delta){
		this->delta+=delta;
	}

	long double calcWeightResult(){
		return this->inputValue*this->weight;
	}
	long double getInput(){
		return this->inputValue;
	}
	int getFrom(){
		return this->fromNeuron;
	}

	void adjustWeights(long double learningRate,long double momentum,bool ei,bool start){
		long double a=this->weight;
		if(ei){
			this->weight += fRand(0.00000001, 0.000000001);
		}
		if(start){
			this->weight += this->errorSumIn;
		}else{
			this->weight += this->errorSumIn+this->bias+(momentum*(this->weight-this->pWeight));
		}
		this->delta=0;
		this->bias=0;
		this->errorSumIn=0;
		this->pWeight=a;
	}

	void backupWeight(){
		this->pWeight=this->weight;
	}

	void rollbackWeight(){
		if(this->pWeight>0){
			this->weight=this->pWeight;
		}
	}
};

class Neuron{
	 public:
	 long double inputSum;
	 long double outputSum;
	 int layer;
	 bool outputLayer;
	 int id;
	 bool func;
	 long double wInput;
	 long double delta;
	 int inputs;
	 long double finalOutputError;
	 long double ao;
	 vector<unique_ptr<Input>> in;
	 unique_ptr<ActivationFunction> af;
	 long double weightSum;
	 long double errorSumAbs;
	 long double currentInputSum;
	 long double currentWeightSum;

	 Neuron(int id,int layer,int inputs,int func,bool pl,bool pl2){
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
		 this->outputLayer=false;
		 if(func==0){
			 this->af=make_unique<ActivationFunctionSoftPlus>();
		 }else if(func==1){
			 if(layer==1){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else if(layer==2){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else if(pl){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else if(pl2){
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }else{
				 this->af=make_unique<ActivationFunctionSoftPlus>();
			 }
		 }else if(func==2){
			 this->outputLayer=true;
			 this->af=make_unique<ActivationFunctionLinear>();
		 }
	 };
	 void setErrorSumAbs(long double es){
		 this->errorSumAbs=es;
	 }

	 int getId(){
		 return this->id;
	 }

	 void aOutput(){
		 long double aol=this->af->activationOutput(this->currentInputSum);
		 this->ao=aol;
	 }
	 void finalOutput(long double expected,int iteration, int olocation,bool showOutput,double& eSum){
		 try{
			 long double es=sqrt(pow(expected-this->ao,2));
			 this->setErrorSumAbs(es);
			 eSum+=es;
			 if(showOutput){
				 cout << "Iteration: " << iteration << " expected at neuron"<< olocation << ": " << expected << " output: "<< this->ao <<endl;
			 }
		 }catch (const std::exception& ex) {
		 			 throw_line(ex.what());
		 			 exit(EXIT_FAILURE);
		 }
	 }

	 long double getOutputStatic(){
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
	 void setInput(int index,int fromCheck,long double input,bool first){
		 try{
				 if(this->in.at(index)->fromNeuron==fromCheck){
					 if(first){
						 this->currentInputSum=0;
						 this->currentWeightSum=0;
						 this->inputSum=0;
						 this->weightSum=0;
					 }
					 if(this->layer==0){
						 this->in.at(index)->setInputUw(input);
					 }else{
						 this->in.at(index)->setInput(input);
					 }
					 this->currentInputSum+=this->in.at(index)->inputValue;
					 this->currentWeightSum+=this->in.at(index)->weight;
				 }else{
					 throw_line("Improper input set");
					 exit(EXIT_FAILURE);
				 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }

	 void outputNeuronCalcError(int inputIndex,unique_ptr<Neuron> & pNeuron,long double lRate,long double momentum,bool eInc,long double bias){
			 try{
				if(this->in.at(inputIndex)->fromNeuron==pNeuron->id){
					if(eInc){lRate*=499;};
					long double delta=this->errorSumAbs*af->dFunction(this->currentInputSum);
					long double error=delta*lRate*pNeuron.get()->ao;
					long double bv=delta*lRate*bias;
					this->in.at(inputIndex)->sumDelta(delta);
					this->in.at(inputIndex)->sumBias(bv);
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
		 void hiddenNeuronCalcError(int inputIndex,int cnliIn,unique_ptr<Neuron> & pNeuron,unique_ptr<Neuron> & nNeuron,long double lRate,long double momentum,bool eInc,long double bias){
			 try{
				 int pNeuronFromNeuron=this->in.at(inputIndex)->fromNeuron;
				 int nFromNeuron=nNeuron.get()->in.at(cnliIn)->fromNeuron;
				 if(pNeuronFromNeuron==pNeuron->id && nFromNeuron==this->id){
					 if(eInc){lRate*=11;};
					 long double delta=fabs((nNeuron.get()->in.at(cnliIn)->delta*this->currentWeightSum*this->af->dFunction(this->currentInputSum))*-1);
					 long double error=delta*lRate*pNeuron->getOutputStatic();
					 long double bv=delta*lRate*bias;
					 this->in.at(inputIndex)->sumDelta(delta);
					 this->in.at(inputIndex)->sumBias(bv);
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


	 long double getDelta(){
		 return this->delta;
	 }

	 void setNewIterationValues(){
		 this->currentInputSum=0;
		 this->currentWeightSum=0;
		 /*
	  	 for(unsigned int c=0;c<this->in.size();c++){
	    	this->in.at(c)->resetValues();
	     }*/
	 }

	 void rollbackWeights(){
		 for(unsigned int c=0;c<this->in.size();c++){
		 	    	this->in.at(c)->rollbackWeight();
		 }
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
	  vector<vector<long double>> inputData;
	  int inputDataLength;
	  vector<vector <long double>> idealData;
	  vector<vector<int>> layers;
	  int layerSize;
	  int idealDataLenght;
	  int neuronMapSize;
	  int neuronMapSizeM;
	  int inputDataSize;
	  int inputDataSizeM;
	  int it;
	  int sample;
	  double totalReturnValue;
	  double totalReturnValueP;
	  double totalReturnValuePR;
	  int mCutoff;
	  double pCutoff;
	  long double totalOutput;
	  long double errorPercentage;
	  long double bestErrorRate;
	  long double currentErrorValue;
	  long double currentOutputSum;
	  double learningRate;
	  double oLearningRate;
	  double maxLearningRate;
	  double minLearningRate;
	  double bias;
	  bool eIncreasing;
	  int eIncreasingCount;
	  bool eStalling;
	  double momentum;
	  double oMomentum;
	  double minMomentum;
	  bool train;
	  vector<unique_ptr<Neuron>> neurons;
	  int neuronsVSize;
	  int neuronsVSizeM;
	  double nDivider;
	  double cutoffErrorRate;
	  int sampleMax;
	  int sampleMin;
	  int aSample;

	  NeuralNetwork(vector<vector<int>> neuronMap,vector<vector<long double>> inputData,vector<vector<long double>> idealData,double learningRate,double momentum,bool train,double cErrorRate,int mcf,double cmcf,int sampleMax,int sampleMin){
		  cout.precision(14);
		  this->neuronMap=neuronMap;
		  this->neuronMapSize=neuronMap.size();
		  this->inputData=inputData;
		  this->inputDataLength=inputData.size();
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
		  this->pCutoff=cmcf;
		  this->sampleMax=sampleMax;
		  this->sampleMin=sampleMin;
		  this->sample=sample;
		  this->minMomentum=this->momentum/10000;
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
		  this->eStalling=false;
		  this->train=train;
		  this->it=0;
		  this->totalReturnValue=0;
		  this->totalReturnValueP=this->totalReturnValue;
		  this->totalReturnValuePR=this->totalReturnValue;
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
		  this->aSample=0;
		  this->bias=0;
	  }

	  void resetNetworkForLoaded(){
		  cout.precision(14);
		  this->inputDataLength=inputData.size();
		  this->idealDataLenght=idealData.size();
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
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

/*	  void resetNetworkForLoaded(){
		  cout.precision(14);
		  this->inputDataLength=inputData.size();
		  this->idealDataLenght=idealData.size();
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
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
	  }*/

	  void createNetwork(){
		  try{
		      if(this->neuronMapSize>0){
		    	 int id=0;
		    	 int sId=0;
		    	 int indx=0;
		    	 int layer=0;
		    	 int func=0;
		    	 bool pl=false;
		    	 bool pl2=false;
				 while(layer<this->neuronMapSize){
					 	     if(layer==1){func++;};
					 	     if(layer==this->neuronMapSizeM){func++;};
					 	 	 for(int i=0;i<neuronMap.at(layer).at(0);i++){
						 	 	 this->neurons.push_back(make_unique<Neuron>(id,layer,neuronMap.at(layer).at(1),func,pl,pl2));
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
					 	 	 if(this->neuronMapSizeM>2){
								 if(layer==this->neuronMapSizeM-2){
									pl2=true;
								 }
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


	  void iterate(){
		while(this->it<this->mCutoff){
				 	int sample=rand() % sampleMax + sampleMin;
				 	this->lRun(false,sample);
		}
	  }

	  void pRun(bool mt){
		 if(mt){

		 }else{
			 for(int dataLocation=0;dataLocation<this->inputDataSize;dataLocation++){
				 this->feedForward(true,true,dataLocation);
			 }
		 }

	  }
	  void lRun(bool mt, int sample){
		  try{
			  	 if(mt){

			  	 }else{
						int rLoc=rand()%(this->inputDataSize-sample-0 + 1) + 0;
						int rLoco=rLoc;
						int cutoff=rLoc+sample;
						long double cycles=20.0;
						long double heat=100;
						long double bias=0;
						bool cn=true;
						int ic=0;
						int icMax=2;
						int ci=0;
						int cMax=10;
						int sti=0;
						int stiMax=2;
						long double rt=((long double)this->it/(long double)this->mCutoff);

							while(ic<icMax && cn){
								while(rLoc<cutoff){
											this->feedForward(false,true,rLoc);
											this->backPropagate(bias);
											rLoc++;
								}
								if(ic==0){
									this->learn(true);
								}else{
									this->learn(false);
								}
								cn=this->checkDataAndCleanUp(sample,false,true,true);
								ic++;
								rLoc=rLoco;

								if(this->eIncreasing){
									bias+=0.49;
								}
							}

							while(sti<stiMax && this->eStalling && cn){
								while(rLoc<cutoff){
											this->feedForward(false,true,rLoc);
											this->backPropagate(bias);
											rLoc++;
								}
								if(sti==0){
									this->learn(true);
								}else{
									this->learn(false);
								}
								cn=this->checkDataAndCleanUp(sample,false,true,true);
								sti++;
								rLoc=rLoco;
								if(this->eIncreasing){
									bias+=0.033333;
								}
							}

							while(ci<cMax && !this->eIncreasing && cn){
								while(rLoc<cutoff){
											this->feedForward(false,true,rLoc);
											this->backPropagate(bias);
											rLoc++;
								}
								if(ci==0){
									this->learn(true);
								}else{
									this->learn(false);
								}
								cn=this->checkDataAndCleanUp(sample,false,true,true);
								ci++;
								rLoc=rLoco;
								if(this->eIncreasing){
									bias+=0.033333;
								}
							}

							if(rt<0.3){
								rLoc=rLoco;
								if(this->totalReturnValuePR > 0.29){
									cycles=32+32*rt;
									heat=1111+(1111*rt);
									this->runAnnealing(rLoc, cutoff, heat, cycles,false,this->totalReturnValueP, sample);
								}else if(this->totalReturnValuePR > 0.09){
									cycles=22+22*rt;
									heat=111+111*rt;
									this->runAnnealing(rLoc, cutoff, heat, cycles,false,this->totalReturnValueP, sample);
								}
							}

/*							if(cn && !this->eIncreasing && this->totalReturnValuePR>0.05){
								rLoc=rLoco;
								cycles=21+21*rt;
								heat=11111+(11111*rt);
								this->runAnnealing(rLoc, cutoff, heat, cycles,false,this->totalReturnValueP, sample);
							}*/



			  	 }

		  }catch (const std::exception& ex) {
				 throw_line(ex.what());
				 exit(EXIT_FAILURE);
		  }

	  }

	  void setLearningRate(long double lr){

	  }
	  void setMomentum(long double sm){

	  }
	  void iLearningRate(long double il){

	  }
	  void iMomentum(long double im){

	  }
	  void dLearningRate(long double dl){

	  }
	  void dMomentum(long double dm){

	  }
	  void dLearningRateGradual(long double limit){

	  }

	  void getNeuronErrorPercentage(Neuron n){

	  }

	  void feedForward(bool showOutput,bool first,int dataLocation){
		 try{
			 if(dataLocation<this->inputDataLength){
				  int oDataLoc=0;
				  int layer;
				  int ifi=0;
				  int dLocc=0;
					 for(int c=0;c<this->neuronsVSize;c++){
							layer=this->neurons.at(c)->layer;
							if(layer==0){
								this->neurons.at(c)->setInput(0,-1, this->inputData.at(dataLocation).at(ifi),true);
								ifi++;
							}else if(layer>0){
									int i=0;
									for(int ix=layers.at(layer-1).at(0);ix<layers.at(layer-1).at(1);ix++){
										this->neurons.at(c)->setInput(i,ix,this->neurons.at(ix)->ao,first);
										first=false;
										i++;
									}
							}
							dLocc++;
							this->neurons.at(c)->aOutput();
							if(layer==this->neuronMapSizeM){
								this->neurons.at(c)->finalOutput(this->idealData.at(dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput,this->totalReturnValue);
								oDataLoc++;
							}
							first=true;
					  }
			 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	  }

	  void backPropagate(long double tbias){
		  try{
			  int cNeuronIndex=this->neuronsVSizeM;
			  tbias+=this->bias;
				  for(int layer=this->neuronMapSizeM;layer>0;layer--){
					  if(layer==this->neuronMapSizeM){
						  do{
							  // Neurons end at the previous layer
							  int previousNeuronId=this->layers.at(layer-1).at(1)-1;

							  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0 && previousNeuronId>-1;pi--,previousNeuronId--){
									  this->neurons.at(cNeuronIndex)->outputNeuronCalcError(pi, this->neurons.at(previousNeuronId), this->learningRate, this->momentum, this->eIncreasing,tbias);
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
										  this->neurons.at(cNeuronIndex)->hiddenNeuronCalcError(pi,nlin,this->neurons.at(previousNeuronId),this->neurons.at(nextLayerNeuronId),this->learningRate,this->momentum,this->eIncreasing,tbias);
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
	  void learn(bool start){
		  int cNeuronIndex=this->neuronsVSizeM;
		  int layer=this->neurons.at(cNeuronIndex)->layer;
		  do{
				  for(int c=0;c<neurons.at(cNeuronIndex)->inputs;c++){
					  this->neurons.at(cNeuronIndex)->in.at(c)->adjustWeights(this->learningRate, this->momentum,this->eIncreasing,start);
				  }
			  	  if(layer==this->neuronMapSizeM){
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void calcFinalError(){
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

	  void rollback(){
			  int cNeuronIndex=this->neuronsVSizeM;
			  int layer=this->neurons.at(cNeuronIndex)->layer;
			  do{
				  	  this->neurons.at(cNeuronIndex)->rollbackWeights();
					  cNeuronIndex--;
					  layer=this->neurons.at(cNeuronIndex)->layer;
			  }while(layer>0);
	  }

	  bool checkDataAndCleanUp(int sample,bool suppress,bool rc,bool brc){
			this->learningRate=this->learningRate/(1.0+((double)this->it/(double)this->pCutoff));
		    if(this->totalReturnValueP>0){
					  if((this->totalReturnValue < this->totalReturnValueP)){
						this->eIncreasing=false;
						this->eIncreasingCount=0;
					  }else if((this->totalReturnValue > this->totalReturnValueP)){
						this->eIncreasing=true;
						this->eIncreasingCount++;
					  }else if((this->totalReturnValue == this->totalReturnValueP)){
						this->eIncreasing=true;
						this->eIncreasingCount++;
						this->eStalling=true;
					  }
		  	 }
			 if(rc){
				 if(this->eIncreasing){
					 this->rollback();
					 if(brc){
						 this->bias+=0.05;
					 }
				 }
			 }

			 this->totalReturnValueP=this->totalReturnValue;
			 this->totalReturnValuePR=this->totalReturnValueP/(double)sample;
			 if(!suppress){
				 cout << "Iteration: "<< this->it << " current deviation from ideal results: "<< this->totalReturnValuePR << endl;
				 this->it++;
			 }
		     this->totalReturnValue=0;
		     if(this->totalReturnValuePR < this->cutoffErrorRate){
		    	 return false;
		     }else{
		    	 return true;
		     }
	  }

	  void runAnnealing(int rLoc, int cutoff,long double heat,long double cycles,bool ca,long double oError,int sample){
			long double currentTemp=heat;
			long double stoptemp=heat*0.01;
			long double pError=oError;
			long double ratio = exp(log(stoptemp / heat) / (cycles - 1));
			bool stop=false;
			long double stopRate=10000;
			long double stopRatet=0.01;
			int eLoc2=rLoc;
			int ai=0;
			int c=0;

			do{
				c=0;
				do{
						this->randomize(currentTemp,heat);
						while(rLoc<cutoff){
								this->feedForward(false,true,rLoc);
								rLoc++;
						}
						rLoc=eLoc2;
						this->checkDataAndCleanUp(sample,true,true,false);
						ai++;
						pError=this->totalReturnValueP;
						cout << "Annealing: "<< ai << " current deviation from ideal results: "<< fabs(this->totalReturnValuePR) << endl;

						stopRate=pError/oError;
						if(stopRate<stopRatet || pError <= this->cutoffErrorRate || this->eIncreasingCount>2){
							stop=true;
						}
						if(ca){
							if(ai>8000000){
								stop=true;
							}
						}
						if(ai>10000000){
							stop=true;
						}
					c++;
				}while(c < cycles && !stop);

				currentTemp *= ratio;

			}while(currentTemp>stoptemp && !stop);
			this->eIncreasingCount=0;
			this->eIncreasing=false;
		}

		void randomize(long double currentTemp,long double startTemp){
			int i=0;
			int ns=this->neurons.size();
			while(i<ns){
				if(this->neurons.at(i)->layer>1){
					int si=this->neurons.at(i)->in.size();
					for(int c=0;c<si;c++){
						long double add = 0.0000005 - fRand(0.0000003,0.0000001);
						add /= startTemp;
						add *= currentTemp;
						this->neurons.at(i)->in.at(c)->backupWeight();
						this->neurons.at(i)->in.at(c)->weight+=add;
					}
				}
				i++;
			}

		}



};

#endif /* NEURAL_H_ */
