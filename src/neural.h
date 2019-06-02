//============================================================================
// Name        : Neuralm.cpp
// Author      : Lauri Turunen
// Version     :
// Copyright   : Lauri Turunen
// Description : Multilayer neural network software.
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
#include <cereal/types/base_class.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <ostream>
#include <numeric>
#include <chrono>
#include <iostream>
#include <iterator>
#include <omp.h>
using namespace std;

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

double fRand( double fMax, double fMin)
{
     double f = ( double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

class ActivationFunction{
	public:
	ActivationFunction(){};
	virtual ~ActivationFunction(){};
	virtual  double activationOutput( double d) = 0;
	virtual  double dFunction( double d) = 0;
};

class ActivationFunctionLinear final: public ActivationFunction{
	public:
	ActivationFunctionLinear(){};
	~ActivationFunctionLinear(){};
	 double activationOutput( double d){
		return d;
	}
	 double dFunction( double d){
		return 1;
	}
};


class ActivationFunctionTanh final: public ActivationFunction{
	public:
	ActivationFunctionTanh(){};
	~ActivationFunctionTanh(){};
	 double activationOutput( double d){
		return (exp(d*2.0)-1.0)/(exp(d*2.0)+1.0);
	}
	 double dFunction( double d){
		return 1.0-pow(activationOutput(d), 2.0);
	}
};

class ActivationFunctionSigmoid final: public ActivationFunction{
	public:
	ActivationFunctionSigmoid(){};
	~ActivationFunctionSigmoid(){};
	 double activationOutput(double d){
		    double e = exp(d)*9;
	       return - 1.0/(1.0+e);
	}
	 double dFunction( double d){
		 double e = exp(d);
		return - e/((1.0+e)*(1.0+e));
	}
};

class ActivationFunctionRectifiedRelu final: public ActivationFunction{
	public:
	ActivationFunctionRectifiedRelu(){};
	~ActivationFunctionRectifiedRelu(){};
	 double activationOutput(double d){
		if(d<0){d=0;}
		return d;
	}
	 double dFunction( double d){
		if(d<0){d=0;}else if(d>=0){d=1;};
		return d;
	}
};

class ActivationFunctionSoftPlus final: public ActivationFunction{
	public:
	ActivationFunctionSoftPlus(){};
	~ActivationFunctionSoftPlus(){};
	 double activationOutput( double d){
		return log(1+pow(d,2));
	}
	 double dFunction( double d){
		return 1/(1+pow(d,2));
	}
};

class ActivationFunctionSin final: public ActivationFunction{
	public:
	ActivationFunctionSin(){};
	~ActivationFunctionSin(){};
	 double activationOutput( double d){
		return sin(d);
	}
	 double dFunction( double d){
		return cos(d);
	}
};


class ActivationFunctionSinc final: public ActivationFunction{
	public:
	ActivationFunctionSinc(){};
	~ActivationFunctionSinc(){};
	 double activationOutput( double d){
		if(d==0){
			return 1;
		}else{
			return (sin(d)/d);
		};
	}
	 double dFunction( double d){
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
	double errorSumIn;
	double delta;
	double weight;
	double pWeight;
	double inputValue;
	double bias;
	Input(int inputId,int layer,int fromNeuron){
		this->id=inputId;
		this->fromLayer=layer-1;
		this->toLayer=layer;
		this->fromNeuron=fromNeuron;
		this->errorSumIn=0.0;
		this->delta=0.0;
		if(layer>0){
			this->weight=0.011+fRand(0.01,0.00088199)+fRand(0.001,0.000089199)+fRand(0.00035,0.0000345199)+fRand(0.00035,0.000349199);
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
	 double setInput(double input){
		double iv=this->weight*input;
		this->inputValue=iv;
		return iv;
	}

	 double setInputUw(double input,double iMax){
		double iv=this->weight*(input/iMax);
		this->inputValue=iv;
		return iv;
	}
	 double getWeight(){
		return this->weight;
	}
	void sumError( double error){
		this->errorSumIn+=error;
	}

	void sumBias( double bv){
			this->bias+=bv;
	}
	void sumDelta( double delta){
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

	void adjustWeights(double learningRate, double momentum,bool ei,bool start){
		double a=this->weight;
		if(ei){
			this->weight +=fRand(0.00000000000000001, 0.0000000000000000089);
		}
		if(start){
			this->weight += this->errorSumIn+this->bias;
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
	 double inputSum;
	 double outputSum;
	 int layer;
	 bool outputLayer;
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
	 double gmInputValue;

	 Neuron(int id,int layer,int inputs,int func,bool pl,bool pl2,double gmInputValue){
		 this->inputSum=0;
		 this->currentInputSum=0;
		 this->outputSum=0;
		 this->layer=layer;
		 this->inputs=inputs;
		 this->func=func;
		 this->gmInputValue=gmInputValue;
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
	 void setErrorSumAbs( double es){
		 this->errorSumAbs=es;
	 }

	 int getId(){
		 return this->id;
	 }

	  double aOutput(){
		 double aol =this->af->activationOutput(this->currentInputSum);
		 this->ao=aol;
		 return aol;
	 }
	 void finalOutput( double expected,int iteration, int olocation,bool showOutput,double& eSum){
		 try{
			 double es=sqrt(pow(expected-this->ao,2));
			 this->setErrorSumAbs(es);
			 eSum+=es;
			 if(showOutput){
				 cout << "Iteration: " << iteration << " expected at neuron"<< olocation << ": " << expected << " output: "<< this->ao <<'\n';
			 }
		 }catch (const std::exception& ex) {
		 			 throw_line(ex.what());
		 			 exit(EXIT_FAILURE);
		 }
	 }

	  double getOutputStatic(){
		 return this->ao;
	 }

	 void setInputs(int & inputId,int inputIdStart,int inputIdEnd){
		 try{
			 if(this->layer>0){
				 int fromNeuron=inputIdStart;
				 for(int i=0;i<this->inputs;++i){
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
	 void setInput(int index,int fromCheck, double input,double iMax){
		 try{
				 if(this->in.at(index)->fromNeuron==fromCheck){
					 if(index==0){
						 this->currentInputSum=0;
						 this->currentWeightSum=0;
						 this->inputSum=0;
						 this->weightSum=0;
					 }
					 if(this->layer==0){
						 this->currentInputSum+=this->in.at(index)->setInputUw(input,iMax);
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

	 void outputNeuronCalcError(int inputIndex,int pNeuronId, double pNeuronOutput, double lRate, double momentum,bool eInc, double bias){
			 try{
				int lFromNeuron=this->in.at(inputIndex)->fromNeuron;
				if(lFromNeuron==pNeuronId){
					if(eInc){lRate*=2.5;};
					 double delta=this->errorSumAbs*af->dFunction(this->currentInputSum);
					 double error=fabs(delta*lRate*pNeuronOutput*-1);
					 double bv=fabs(delta*lRate*bias*-1);
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
		 void hiddenNeuronCalcError(int inputIndex,int nNeuronInputNeuronId, double cnliInDelta, double pNeuronOutput,int pNeuronId, double lRate, double momentum,bool eInc, double bias){
			 try{
				 int pNeuronFromNeuron=this->in.at(inputIndex)->fromNeuron;
				 if(pNeuronFromNeuron==pNeuronId && nNeuronInputNeuronId==this->id){
					 if(eInc){lRate*=1.89;};
					  double delta=fabs(cnliInDelta*this->currentWeightSum*this->af->dFunction(this->currentInputSum)*-1);
					  double error=fabs(delta*lRate*pNeuronOutput*-1);
					  double bv=fabs(delta*lRate*bias*-1);
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


	  double getDelta(){
		 return this->delta;
	 }

	 void rollbackWeights(){
		 for(unsigned int c=0;c<this->in.size();c++){
		 	    	this->in.at(c)->rollbackWeight();
		 }
	 }

};

struct dataSet
{
  bool   b;
  double d;

  template <class Archive>
  void serialize( Archive & ar )
  {
    ar( b, d );
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
	 int sampleMax=20;
	 int sampleMin=3;
	 int m=500;
	 vector<vector<int>> neuralMap;
	 vector<vector<double>>inputData;
	 vector<vector<double>>idealData;
	 vector< double> weights;
	 int neuronMapSize;
	 int neuronMapSizeM;
	 int inputDataSize;
	 int inputDataSizeM;
	 int idealDataSize;
	 int idealDataSizeM;
	 double maxInputValue;
	 vector<int> neuronsList={3,20,20,20,20,1};

	 void init(){
		 try{
			 this->nNeurons=neuronsList.size();
			 int f=0;
			 int s=0;
			 for(int a = 0; a < this->nNeurons; ++a){
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

	 void setData(vector<vector< double>>inputData,vector<vector< double>>idealData){
		 if(inputData.size()>0 && idealData.size()>0){
			 this->inputData=inputData;
			 this->idealData=idealData;
			 this->setInputDataMax();
		 }
	 }

	 void generateTrainingData(){
		 this->inputData.push_back({0.101,0.16152,0.11196});
		 this->inputData.push_back({0.108,0.13089,0.13335});
		 for(int u=2;u<this->m;++u){
			  double ud=( double)u;
			  double d1=abs(sin(((this->inputData.at(u-1).at(0)+this->inputData.at(u-2).at(0)+(0.1/ud))/2)+((0.01/(ud/this->m))*fRand(0.1, 0.005))+ud*0.01));
			  double d2=abs(sin(((this->inputData.at(u-1).at(1)+this->inputData.at(u-2).at(1)+(0.1/ud))/2)+((0.01/(ud/this->m))*fRand(0.07, 0.001))+ud*0.011));
			  double d3=abs(sin(((this->inputData.at(u-1).at(2)+this->inputData.at(u-2).at(2)+(0.1/ud))/2)+((0.01/(ud/this->m))*fRand(0.11, 0.007))+ud*0.0111));
			 this->inputData.push_back({d1,d2,d3});
		 }
		 int iSize=this->inputData.size();

		 for(int t=0;t<iSize;++t){
			  double ival=((this->inputData.at(t).at(0)+this->inputData.at(t).at(1)+this->inputData.at(t).at(2))/3);
			 this->idealData.push_back({
				 ival
			 });
		 }
		 this->inputDataSize=this->inputData.size();
		 this->inputDataSizeM=this->inputDataSize-1;
		 this->idealDataSize=this->idealData.size();
		 this->idealDataSizeM=this->idealDataSize-1;
		 this->setInputDataMax();
	 }

	 void setInputDataMax(){
		 try{
			 if(!this->inputData.empty()){
				 double max=0;
				 for(unsigned int i=0;i<this->inputData.size();++i){
					 int is=this->inputData.at(i).size();
						 for(int ii=0;ii<is;++ii){
							 double lm=this->inputData.at(i).at(ii);
								 if(lm>max){
									 max=lm;
								 }
						 }
				 }
				 if(max==0){
					 throw_line("Max cannot be zero");
				 }
				 this->maxInputValue=max;
			 }
		 }catch (const std::exception& ex) {
		 	throw_line(ex.what());
		 	exit(EXIT_FAILURE);
		 }
	 }

	 void setInputWeight( double weight){
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
	  double totalOutput;
	  double errorPercentage;
	  double bestErrorRate;
	  double currentErrorValue;
	  double currentOutputSum;
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
					 	     if(layer==1){++func;};
					 	     if(layer==this->skeleton.neuronMapSizeM){++func;};
					 	 	 for(int i=0;i<this->skeleton.neuralMap.at(layer).at(0);++i){
						 	 	 this->neurons.push_back(make_unique<Neuron>(id,layer,this->skeleton.neuralMap.at(layer).at(1),func,pl,pl2,this->skeleton.maxInputValue));
						 	 	 ++id;
						 	 	 ++indx;
					 	 	 }
					 	 	 layers.push_back({sId,id,indx});
					 	 	 indx=0;
					 	 	 sId=id;
					 	 	 ++layer;
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
				 cout << "Network created "<< '\n';
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
			 for(int prin=0;prin<this->neuronsVSize;++prin){
				  int lSize=this->neurons.at(prin)->in.size();
				  for(int li=0;li<lSize;++li){
					  this->neurons.at(prin)->in.at(li)->weight=skeleton.weights.at(wCount);
					  ++wCount;
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

	  void pRun(){
			 for(int dataLocation=0;dataLocation<this->skeleton.inputDataSize;dataLocation++){
				 this->feedForward(true,dataLocation);
			 }
	  }
	  void lRun(bool mt, int sample){
		  try{
			  	  	  	double rt=(double)this->it/(double)this->skeleton.mCutoff;
			  	  	  	// double art=(double)this->skeleton.mCutoff/(double)this->it;
						int rLoc=rand()%(this->skeleton.inputDataSize-sample-0 + 1) + 0;
						int cutoff=rLoc+sample;
						double bias=5000*rt;
						bool cn=true;
						int siMax=3;

						for(int si=0;si<siMax && cn;++si){
							runTrainingRound(rLoc,cutoff,bias,sample,false,true);
							cn=this->checkDataAndCleanUp(sample,false,true,true);
							if(this->eIncreasing && si>0){
								bias+=this->bias;
							}
						}

						if(this->nCycle>11000 || this->nCycle==0){
							int mc=this->skeleton.inputDataSize;
							int biMax=2;
							for(int bi=0;bi<biMax;++bi){
								runTrainingRound(0,mc,bias,mc,true,true);
								this->checkDataAndCleanUp(mc,false,true,false);
							}
							this->nCycle=0;
						}
						this->nCycle++;

		  }catch (const std::exception& ex) {
				 throw_line(ex.what());
				 exit(EXIT_FAILURE);
		  }

	  }

	  void runTrainingRound(int rLoc,int cutoff, double bias,int sample,bool ls,bool stochastic){
				for(int r=rLoc;r<cutoff;++r){
					this->feedForward(false,r);
					this->backPropagate(bias,sample,ls);
					if(stochastic && r>rLoc){
						this->learn(false);
					}else{
						this->learn(true);
					}
				}

	  }

	  void runAnnealingTrainingRound(int cf, double cycles, double heat){
			this->runAnnealing(0, cf, heat, cycles,false);
	  }

	  void feedForward(bool showOutput,int dataLocation){
			 if(dataLocation<this->skeleton.inputDataSize){
					try{
					    int oDataLoc=0;
					    int ifi=0;
					    int layer=0;
							  do{
									  for(int c=this->layers.at(layer).at(0);c<this->layers.at(layer).at(1);++c){
											if(layer==0){
												this->neurons.at(c)->setInput(0,-1, this->skeleton.inputData.at(dataLocation).at(ifi),this->skeleton.maxInputValue);
												ifi++;
											}else if(layer>0){
												int i=0;
												for(int ix=layers.at(layer-1).at(0);ix<layers.at(layer-1).at(1);++ix){
													this->neurons.at(c)->setInput(i,ix,this->neurons.at(ix)->getOutputStatic(),this->skeleton.maxInputValue);
													++i;
												}
											}
											this->neurons.at(c)->aOutput();
											if(layer==this->skeleton.neuronMapSizeM){
												  this->neurons.at(c)->finalOutput(this->skeleton.idealData.at(dataLocation).at(oDataLoc),this->it,oDataLoc,showOutput,this->totalReturnValue);
												  ++oDataLoc;
											}
									  }
									  ++layer;
							  }while(layer<this->skeleton.neuronMapSize);

					 }catch (const std::exception& ex) {
						 throw_line(ex.what());
						 exit(EXIT_FAILURE);
					 }
			 }
	  }

	  void backPropagate( double tbias,int sample,bool ls){
		  try{
			  int cNeuronIndex=this->neuronsVSizeM;
			  double alr=this->skeleton.learningRate;
			  if(ls){
				  alr*=0.0002591;
			  }else{
				  alr*=(( double)sample/( double)skeleton.inputDataSize)*1.3333;
			  }

				  for(int layer=this->skeleton.neuronMapSizeM;layer>0;--layer){
					  if(layer==this->skeleton.neuronMapSizeM){
						  do{
								  int previousNeuronId=this->layers.at(layer-1).at(1)-1;
								  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0;--pi){
									  if(previousNeuronId>-1){
											  int pNeuronId=this->neurons.at(previousNeuronId)->id;
											  double pNeuronOutput=this->neurons.at(previousNeuronId)->getOutputStatic();
											  this->neurons.at(cNeuronIndex)->outputNeuronCalcError(pi,pNeuronId,pNeuronOutput,alr,this->skeleton.momentum,this->eIncreasing,tbias);
											  --previousNeuronId;
									  }
								  };

							 // Move on to next neuron
							--cNeuronIndex;
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
								  for(int pi=this->layers.at(layer-1).at(2)-1;pi>=0;--pi){
										  if(previousNeuronId>-1){
											  int pNeuronId=this->neurons.at(previousNeuronId)->id;
											   double pNeuronOutput=this->neurons.at(previousNeuronId)->getOutputStatic();
											   double nNeuronDelta=this->neurons.at(nextLayerNeuronId)->in.at(nlin)->delta;
											  int nNeuronId=this->neurons.at(nextLayerNeuronId)->in.at(nlin)->fromNeuron;
											  this->neurons.at(cNeuronIndex)->hiddenNeuronCalcError(pi,nNeuronId,nNeuronDelta,pNeuronOutput,pNeuronId,alr,this->skeleton.momentum,this->eIncreasing,tbias);
											  --previousNeuronId;
										  }
								  };

								  --nextLayerNeuronId;
								  ++nLayerCount;
							  }while(nLayerCount<neuronsInNextLayer);

							  --nlin;
							 // Move on to next neuron
							 --cNeuronIndex;
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
				  for(int c=0;c<neurons.at(cNeuronIndex)->inputs;++c){
					  this->neurons.at(cNeuronIndex)->in.at(c)->adjustWeights(this->skeleton.learningRate, this->skeleton.momentum,this->eIncreasing,start);
				  }
			  	  if(layer==this->skeleton.neuronMapSizeM){
					 this->neurons.at(cNeuronIndex)->errorSumAbs=0;
			  	  }
			  	  --cNeuronIndex;
				  layer=this->neurons.at(cNeuronIndex)->layer;
		  }while(layer>0);

	  }

	  void calcFinalError(){
		  cout << "Iteration: "<< this->it << " Final deviation from ideal results "<< this->totalReturnValue << '\n';
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

	  bool checkDataAndCleanUp(int sample,bool suppress,bool rc,bool ib){
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
					 if(ib){
						 this->bias+=0.01;
					 }
				 }
			 }
			 this->totalReturnValueP=this->totalReturnValue;
			 this->totalReturnValuePR=this->totalReturnValueP/(double)sample;
			 if(!suppress){
				 this->it++;
				 printf("Iteration: %i current deviation from ideal results: %.16g  \n",this->it,this->totalReturnValuePR);
			 }
		     this->totalReturnValue=0;
		     if(this->totalReturnValuePR < this->skeleton.av){
		    	 return false;
		     }else{
		    	 return true;
		     }
	  }

	  void runAnnealing(int rLoc, int cutoff, double heat, double cycles,bool ca){
			double currentTemp=heat;
			double previusTemp=heat;
			double stoptemp=heat*0.01;
			double pError=this->totalReturnValueP;
			double oError=pError;
			double ratio = exp(log(stoptemp / heat) / (cycles - 1));
			bool stop=false;
			double stopRate=10000;
		    double stopRatet=0.01;
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
						++ai;
						pError=this->totalReturnValueP;
						printf("Annealing: %i current deviation from ideal results: %.16g  \n",ai,this->totalReturnValuePR);

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
					++c;
				}while(c < cycles && !stop);

				previusTemp=currentTemp;
				currentTemp *= ratio;

			}while(currentTemp>stoptemp && !stop);
			this->eIncreasingCount=0;
			this->eIncreasing=false;
		}

		void randomize( double currentTemp, double previusTemp, double startTemp){
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
						 double add = 0.0000005 - fRand(0.0000003,0.0000001);
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
