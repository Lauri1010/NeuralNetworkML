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
#include <cereal/types/base_class.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>

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
	int id;
	int fromNeuron;
	int fromLayer;
	int toLayer;
	long double errorSumIn;
	long double delta;
	long double weight;
	long double pWeight;
	long double inputValue;
	long double bias;
	Input(int inputId,int layer,int fromNeuron){
		this->id=inputId;
		this->fromLayer=layer-1;
		this->toLayer=layer;
		this->fromNeuron=fromNeuron;
		this->errorSumIn=0.0;
		this->delta=0.0;
		if(layer>0){
			this->weight=0.01+fRand(0.0065,0.006388)+fRand(0.0035,0.003288);
		}else{
			this->weight=1;
		}
		this->pWeight= this->weight;
		this->inputValue=0.0;
		this->bias=0.00;
	}
	void resetValues(){
		this->errorSumIn=0;
		this->inputValue=0;
	}
	long double setInput(long double input){
		long double iv=this->weight*input;
		this->inputValue=iv;
		return iv;
	}

	long double setInputUw(long double input){
/*		this->weight=input/(((input-1)/input)*-1)*(1+input);*/
		long double iv=this->weight*input;
		this->inputValue=iv;
		return iv;
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
			this->weight +=fRand(0.000000000001, 0.00000000000001);
		}
		if(start){
			this->weight += this->errorSumIn;
		}else{
			this->weight += this->errorSumIn+this->bias+momentum*(this->weight-this->pWeight);
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

	 long double aOutput(){
		 long double aol=this->af->activationOutput(this->currentInputSum);
		 this->ao=aol;
		 return aol;
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

	 void setInputs(int & inputId,int inputIdStart,int inputIdEnd){
		 try{
			 if(this->layer>0){
				 int fromNeuron=inputIdStart;
				 for(int i=0;i<this->inputs;i++){
						this->in.push_back(make_unique<Input>(inputId,this->layer,fromNeuron));
						fromNeuron++;
				 }
			 }else if(this->layer==0){
				 this->in.push_back(make_unique<Input>(inputId,this->layer,-1));
			 }
			 inputId++;
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	 }
	 void setInput(int index,int fromCheck,long double input){
		 try{
				 if(this->in.at(index)->fromNeuron==fromCheck){
					 if(index==0){
						 this->currentInputSum=0;
						 this->currentWeightSum=0;
						 this->inputSum=0;
						 this->weightSum=0;
					 }
					 if(this->layer==0){
						 this->currentInputSum+=this->in.at(index)->setInputUw(input);
					 }else{
						 this->currentInputSum+=this->in.at(index)->setInput(input);
					 }
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
					if(eInc){lRate*=9.6;};
					long double delta=this->errorSumAbs*0.49*af->dFunction(this->currentInputSum);
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
					 if(eInc){lRate*=4.35;};
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

	 void rollbackWeights(){
		 for(unsigned int c=0;c<this->in.size();c++){
		 	    	this->in.at(c)->rollbackWeight();
		 }
	 }

};

struct NeuralSkeleton{
	 public:
	 int nNeurons;
	 unsigned int nWeights=0;
	 double learningRate=0.00000001111;
	 double momentum=0.79;
	 int mCutoff=80000;
	 double aCutoff=10000;
	 bool train=true;
	 double av=0.015;
	 int sampleMax=150;
	 int sampleMin=10;
	 vector<vector<int>> neuralMap;
	 vector<vector<long double>>inputData;
	 vector<vector<long double>>idealData;
	 vector<long double> weights;
	 int neuronMapSize;
	 int neuronMapSizeM;
	 int inputDataSize;
	 int inputDataSizeM;
	 int idealDataSize;
	 int idealDataSizeM;
	 vector<int> neuronsList={3,20,20,20,20,1};

	 void init(){
		 try{
			 this->nNeurons=neuronsList.size();
			 int f=0;
			 int s=0;
			 for(int a = 0; a < this->nNeurons; a++){
				 if(a>0){
					 f=this->neuronsList.at(a);
					 s=this->neuronsList.at(a-1);
					 this->neuralMap.push_back({f,s});
					 this->nWeights+=f*s;
				 }else if(a==0){
					 f=this->neuronsList.at(a);
					 this->nWeights+=f;
					 this->neuralMap.push_back({f,1});
				 }
			 }
			 this->neuronMapSize=neuralMap.size();
			 this->neuronMapSizeM=this->neuronMapSize-1;
			 this->inputDataSize=0;
			 this->inputDataSizeM=0;
			 this->idealDataSize=0;
			 this->idealDataSizeM=0;
		 }catch (const std::exception& ex) {
		 	throw_line(ex.what());
		 	exit(EXIT_FAILURE);
		 }
	 };

	 void setData(vector<vector<long double>>inputData,vector<vector<long double>>idealData){
		 if(inputData.size()>0 && idealData.size()>0){
			 this->inputData=inputData;
			 this->idealData=idealData;
		 }
	 }

	 void generateTrainingData(){
		 int m=4000;
		 this->inputData.push_back({0.101,0.16152,0.11196});
		 this->inputData.push_back({0.108,0.13089,0.13335});
		 for(int u=2;u<m;u++){
			 long double ud=(long double)u;
			 long double d1=abs(sin(((this->inputData.at(u-1).at(0)+this->inputData.at(u-2).at(0)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.1, 0.005))+ud*0.01));
			 long double d2=abs(sin(((this->inputData.at(u-1).at(1)+this->inputData.at(u-2).at(1)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.07, 0.001))+ud*0.011));
			 long double d3=abs(sin(((this->inputData.at(u-1).at(2)+this->inputData.at(u-2).at(2)+(0.1/ud))/2)+((0.01/(ud/m))*fRand(0.11, 0.007))+ud*0.0111));
			 this->inputData.push_back({d1,d2,d3});
		 }
		 int iSize=this->inputData.size();

		 for(int t=0;t<iSize;t++){
			 long double ival=((this->inputData.at(t).at(0)+this->inputData.at(t).at(1)+this->inputData.at(t).at(2))/3);
			 this->idealData.push_back({
				 ival
			 });
		 }
		 this->inputDataSize=this->inputData.size();
		 this->inputDataSizeM=this->inputDataSize-1;
		 this->idealDataSize=this->idealData.size();
		 this->idealDataSizeM=this->idealDataSize-1;
	 }

	 void setInputWeight(long double weight){
		 try{
			 this->weights.push_back(weight);
		 }catch (const std::exception& ex) {
		 	throw_line(ex.what());
		 	exit(EXIT_FAILURE);
		 }
	 }

	 bool validateNetwork(){
		 try{
			 unsigned int wSize=this->weights.size();
			 if(this->nWeights==wSize){
				 return true;
			 }else{
		    	 throw_line("Improperly constructed neural network, number of weights is incorrect!");
		    	 exit(EXIT_FAILURE);
			 }
		 }catch (const std::exception& ex) {
		 	throw_line(ex.what());
		 	exit(EXIT_FAILURE);
		 }
	 }

	 template <class Archive>
	 void save( Archive & ar ) const
	 {
		ar(this->nNeurons,this->nWeights,this->learningRate,this->momentum,this->mCutoff,this->aCutoff,this->aCutoff,this->av,this->sampleMax,this->sampleMin,this->neuralMap,this->inputData,this->idealData,this->weights,this->neuronMapSize,this->neuronMapSizeM,this->inputDataSize,this->inputDataSizeM,this->idealDataSize,this->idealDataSizeM,this->neuronsList);
	 };

	 template <class Archive>
	 void load( Archive & ar )
	 {
		ar(this->nNeurons,this->nWeights,this->learningRate,this->momentum,this->mCutoff,this->aCutoff,this->aCutoff,this->av,this->sampleMax,this->sampleMin,this->neuralMap,this->inputData,this->idealData,this->weights,this->neuronMapSize,this->neuronMapSizeM,this->inputDataSize,this->inputDataSizeM,this->idealDataSize,this->idealDataSizeM,this->neuronsList);
	 }
};

/**
 Feed forward: the output of each neuron in layer is passed to the input of the next layer neurons
 In backpropagation the error is calculated for each input
 In learning phase the weights are adjusted based on the error data (from output dissiminated to each layer)
 */
class NeuralNetwork{
	  public:
	  NeuralSkeleton skeleton;
	  vector<vector<int>> layers;
	  int layerSize;
	  int it;
	  double totalReturnValue;
	  double totalReturnValueP;
	  double totalReturnValuePR;
	  long double totalOutput;
	  long double errorPercentage;
	  long double bestErrorRate;
	  long double currentErrorValue;
	  long double currentOutputSum;
	  double bias;
	  bool eIncreasing;
	  int eIncreasingCount;
	  bool eStalling;
	  vector<unique_ptr<Neuron>> neurons;
	  int neuronsVSize;
	  int neuronsVSizeM;
	  double nDivider;
	  int nCycle;

	  NeuralNetwork(NeuralSkeleton nSkeleton){
		  this->skeleton=nSkeleton;
		  this->eIncreasing=false;
		  this->eIncreasingCount=0;
		  this->eStalling=false;
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
		  this->neuronsVSize=0;
		  this->neuronsVSizeM=0;
		  this->layerSize=0;
		  this->bias=0;
		  this->nCycle=0;
		  cout.precision(14);
	  }

	  void createNetwork(){
		  try{
		      if(this->skeleton.neuronMapSize>0){
		    	 int id=0;
		    	 int sId=0;
		    	 int indx=0;
		    	 int layer=0;
		    	 int func=0;
		    	 bool pl=false;
		    	 bool pl2=false;
				 while(layer<this->skeleton.neuronMapSize){
					 	     if(layer==1){func++;};
					 	     if(layer==this->skeleton.neuronMapSizeM){func++;};
					 	 	 for(int i=0;i<this->skeleton.neuralMap.at(layer).at(0);i++){
						 	 	 this->neurons.push_back(make_unique<Neuron>(id,layer,this->skeleton.neuralMap.at(layer).at(1),func,pl,pl2));
					 			 id++;
					 			 indx++;
					 	 	 }
					 	 	 layers.push_back({sId,id,indx});
					 	 	 indx=0;
					 	 	 sId=id;
					 	 	 layer++;
					 	 	 if(layer==this->skeleton.neuronMapSizeM-1){
					 	 		pl=true;
					 	 	 }
					 	 	 if(this->skeleton.neuronMapSizeM>2){
								 if(layer==this->skeleton.neuronMapSizeM-2){
									pl2=true;
								 }
					 	 	 }
				 }
			     this->neuronsVSize=this->neurons.size();
			     this->neuronsVSizeM=this->neuronsVSize-1;
			     this->layerSize=this->layers.size();
			     int inputId=0;
			     for(auto &n : neurons){
			    	int lr=n->layer;
			    	if(lr>0){lr-=1;};
			    	n->setInputs(inputId,layers.at(lr).at(0), layers.at(lr).at(1));
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

	  void setWeights(){
		  try{
		     int wCount=0;
			 for(int prin=0;prin<this->neuronsVSize;prin++){
				  int lSize=this->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;li++){
					  this->neurons.at(prin)->in.at(li)->weight=skeleton.weights.at(wCount);
					  wCount++;
				  }
			 }
		   }catch (const std::exception& ex) {
			     throw_line(ex.what());
				 exit(EXIT_FAILURE);
		   }

	  }


	  void iterate(){
		while(this->it<this->skeleton.mCutoff){
				 	int sample=rand() % this->skeleton.sampleMax + this->skeleton.sampleMin;
				 	this->lRun(false,sample);
		}
	  }

	  void pRun(bool mt){
		 if(mt){

		 }else{
			 for(int dataLocation=0;dataLocation<this->skeleton.inputDataSize;dataLocation++){
				 this->feedForward(true,dataLocation);
			 }
		 }

	  }
	  void lRun(bool mt, int sample){
		  try{
			  	 if(mt){

			  	 }else{
						int rLoc=rand()%(this->skeleton.inputDataSize-sample-0 + 1) + 0;
						int cutoff=rLoc+sample;
						long double bias=0;
						bool cn=true;
						int ic=0;
						int icMax=3;
						int ci=0;
						int cMax=14;
						int sti=0;
						int stiMax=3;
						int si=0;
						int siMax=5;
						int li=0;
						int liMax=3;

						while(ic<icMax && cn){
							runTrainingRound(rLoc,cutoff,bias,sample,false,false);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(ic==0){
								this->learn(true);
							}else{
								this->learn(false);
							}
							if(this->eIncreasing){
										bias+=0.1333;
							}
							ic++;
						}

						while(sti<stiMax && this->eIncreasing && cn){
							runTrainingRound(rLoc,cutoff,bias,sample,false,false);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(sti==0){
								this->learn(true);
							}else{
								this->learn(false);
							}
							sti++;
							if(this->eIncreasing){
								bias+=0.1333;
							}
						}

						while(ci<cMax && !this->eIncreasing && cn){
							runTrainingRound(rLoc,cutoff,bias,sample,false,false);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(ci==0){
								this->learn(true);
							}else{
								this->learn(false);
							}
							ci++;
							if(this->eIncreasing){
								bias+=0.1333;
							}
						}

						while(si<siMax && cn){
							runTrainingRound(rLoc,cutoff,bias,sample,false,true);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(this->eIncreasing){
								bias+=0.1333;
							}
							si++;
						}

						while(li<liMax && cn){
							runTrainingRound(rLoc,cutoff,bias,sample,false,false);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(li==0){
								this->learn(true);
							}else{
								this->learn(false);
							}
							if(this->eIncreasing){
										bias+=0.1333;
							}
							li++;
						}


						if(this->nCycle>150 || this->nCycle==0){
							double rt=(double)this->it/(double)this->skeleton.mCutoff;
							int mc=this->skeleton.inputDataSize;
							int bi=0;
							int biMax=5;
							while(bi<biMax){
								runTrainingRound(0,mc,bias,mc,true,true);
								this->checkDataAndCleanUp(mc,false,true,true);
								bi++;
							}
							if(rt<0.7){
								this->runAnnealingTrainingRound(mc,21,111111111);
							}
							this->nCycle=0;
						}

						this->nCycle++;
			  	 }

		  }catch (const std::exception& ex) {
				 throw_line(ex.what());
				 exit(EXIT_FAILURE);
		  }

	  }

	  void runTrainingRound(int rLoc,int cutoff,long double bias,int sample,bool ls,bool stochastic){
			while(rLoc<cutoff){
				this->feedForward(false,rLoc);
				this->backPropagate(bias,sample,ls);
				if(stochastic){
					this->learn(true);
				}
				rLoc++;
			}

	  }

	  void runAnnealingTrainingRound(int cf,long double cycles,long double heat){
			this->runAnnealing(0, cf, heat, cycles,false);
	  }

	  void feedForward(bool showOutput,int dataLocation){
		 try{
			 if(dataLocation<this->skeleton.inputDataSize){
				  int oDataLoc=0;
				  int layer;
				  int ifi=0;
				  int dLocc=0;
					 for(int c=0;c<this->neuronsVSize;c++){
							layer=this->neurons.at(c)->layer;
							if(layer==0){
								this->neurons.at(c)->setInput(0,-1, this->skeleton.inputData.at(dataLocation).at(ifi));
								ifi++;
							}else if(layer>0){
									int i=0;
									for(int ix=layers.at(layer-1).at(0);ix<layers.at(layer-1).at(1);ix++){
										this->neurons.at(c)->setInput(i,ix,this->neurons.at(ix)->getOutputStatic());
										i++;
									}
							}
							dLocc++;
							this->neurons.at(c)->aOutput();
							if(layer==this->skeleton.neuronMapSizeM){
								this->neurons.at(c)->finalOutput(this->skeleton.idealData.at(dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput,this->totalReturnValue);
								oDataLoc++;
							}
					  }
			 }
		 }catch (const std::exception& ex) {
			 throw_line(ex.what());
			 exit(EXIT_FAILURE);
		 }
	  }

	  void backPropagate(long double tbias,int sample,bool ls){
		  try{
			  int cNeuronIndex=this->neuronsVSizeM;
			  tbias+=this->bias;
			  long double alr=this->skeleton.learningRate;
			  if(ls){
				  alr*=0.0000001;
			  }else{
				  alr*=((long double)sample/(long double)skeleton.inputDataSize)*1.3333;
			  }
				  for(int layer=this->skeleton.neuronMapSizeM;layer>0;layer--){
					  if(layer==this->skeleton.neuronMapSizeM){
						  do{
							  // Neurons end at the previous layer
							  int previousNeuronId=this->layers.at(layer-1).at(1)-1;

							  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0 && previousNeuronId>-1;pi--,previousNeuronId--){
									  this->neurons.at(cNeuronIndex)->outputNeuronCalcError(pi, this->neurons.at(previousNeuronId), alr, this->skeleton.momentum, this->eIncreasing,tbias);
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
										  this->neurons.at(cNeuronIndex)->hiddenNeuronCalcError(pi,nlin,this->neurons.at(previousNeuronId),this->neurons.at(nextLayerNeuronId),alr,this->skeleton.momentum,this->eIncreasing,tbias);
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
					  this->neurons.at(cNeuronIndex)->in.at(c)->adjustWeights(this->skeleton.learningRate, this->skeleton.momentum,this->eIncreasing,start);
				  }
			  	  if(layer==this->skeleton.neuronMapSizeM){
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
				  cNeuronIndex--;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void calcFinalError(){
		  cout << "Iteration: "<< this->it << " Final deviation from ideal results "<< this->totalReturnValue << endl;
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
		  	this->skeleton.learningRate=this->skeleton.learningRate/(1.0+((double)this->it/(double)this->skeleton.aCutoff));
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
						 this->bias+=0.00001;
					 }
				 }
			 }

			 this->totalReturnValueP=this->totalReturnValue;
			 this->totalReturnValuePR=this->totalReturnValueP/(double)sample;
			 if(!suppress){
				 this->it++;
				 cout << "Iteration: "<< this->it << " current deviation from ideal results: "<< this->totalReturnValuePR << endl;
			 }
		     this->totalReturnValue=0;
		     if(this->totalReturnValuePR < this->skeleton.av){
		    	 return false;
		     }else{
		    	 return true;
		     }
	  }

	  void runAnnealing(int rLoc, int cutoff,long double heat,long double cycles,bool ca){
			long double currentTemp=heat;
			long double previusTemp=heat;
			long double stoptemp=heat*0.01;
			long double pError=this->totalReturnValueP;
			long double oError=pError;
			long double ratio = exp(log(stoptemp / heat) / (cycles - 1));
			bool stop=false;
			long double stopRate=10000;
			long double stopRatet=0.01;
			int eLoc2=rLoc;
			int ai=0;
			int c=0;
			bool cn=true;

			do{
				c=0;
				do{
						this->randomize(currentTemp,previusTemp,heat);
						while(rLoc<cutoff){
								this->feedForward(false,rLoc);
								rLoc++;
						}
						rLoc=eLoc2;
						cn=this->checkDataAndCleanUp(cutoff,true,true,false);
						ai++;
						pError=this->totalReturnValueP;
						cout << "Annealing: "<< ai << " current deviation from ideal results: "<< this->totalReturnValuePR << endl;

						stopRate=pError/oError;
						if(stopRate<stopRatet || pError <= this->skeleton.av || !cn){
							stop=true;
						}
						if(ca){
							if(ai>100000){
								stop=true;
							}
						}
						if(ai>10000000){
							stop=true;
						}
					c++;
				}while(c < cycles && !stop);

				previusTemp=currentTemp;
				currentTemp *= ratio;

			}while(currentTemp>stoptemp && !stop);
			this->eIncreasingCount=0;
			this->eIncreasing=false;
		}

		void randomize(long double currentTemp,long double previusTemp,long double startTemp){
			int i=0;
			int ns=this->neurons.size();
			double rProb=0;
			while(i<ns){
				if(this->neurons.at(i)->layer>1){
					int si=this->neurons.at(i)->in.size();
					if(this->eIncreasing){
						double r = ((double) rand() / (RAND_MAX)) + 1;
						rProb=exp((currentTemp-previusTemp)/currentTemp);
						if(rProb>r){
							currentTemp=previusTemp;
							this->rollback();
						}
					}
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
