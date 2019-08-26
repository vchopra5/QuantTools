//
//  main.cpp
//  Misc
//
//  Created by Vishal Chopra on 2019-07-26.
//  Copyright © 2019 Vishal Chopra. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cassert>
#include <cmath>


using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

// Structure for each connection in between each neuron
struct Connection{
    double weight;
    double deltaWeight;
};


// Class for each neuron in the neural net
class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned cIndex);
    void setOutputVal(double val){
        m_outputVal = val;
    }
    double getOutputVal(void) const {
        return m_outputVal;
    }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    double sumDOW(const Layer & nextLayer) const;
    static double randomWeight(void){
        return rand()/double(RAND_MAX);
    }
    
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned index;
    double gradient;
};

double Neuron::eta = 0.05;
double Neuron::alpha = 0.75;


double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0;
    
    for(int i=0; i<nextLayer.size() -1; i++){
        sum += m_outputWeights[i].weight * nextLayer[i].gradient;
    }
    
    return sum;
}


Neuron::Neuron(unsigned numOutputs, unsigned cIndex){
    for(int i=0; i<=numOutputs; i++){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    
    index = cIndex;
}

//Using tanh

double Neuron::transferFunction(double x){
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    return (1 - x*x);
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;
    
    for (int i=0; i<prevLayer.size(); i++){
        sum += prevLayer[i].getOutputVal() * prevLayer[i].m_outputWeights[index].weight;
    }
    
    m_outputVal = Neuron::transferFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    gradient = delta*Neuron::transferFunctionDerivative(m_outputVal);
}


void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    
    gradient = dow*Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer &prevLayer){
    for( int i =0; i<prevLayer.size(); i++){
        Neuron &neuron = prevLayer[i];
        double oldDeltaWeight = neuron.m_outputWeights[index].deltaWeight;
        
        double newDeltaWeight = eta*neuron.getOutputVal()*gradient + alpha*oldDeltaWeight;
        
        neuron.m_outputWeights[index].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[index].weight += newDeltaWeight;
        
    }
}

//Overall class for the neural net
class NeuralNet{
public:
    NeuralNet(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputValues);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
 
private:
    vector<Layer> layers;
    double error;
    double averageError;
    double smoothing;
    
};


NeuralNet::NeuralNet(const vector<unsigned> &topology){
    int netSize = (int) topology.size();
    
    for(int i=0; i<netSize; i++){
        layers.push_back(Layer());
        
        unsigned numOutputs = 0;
        
        if (i != topology.size() - 1)
            numOutputs = topology[i-1];
        
        for(int j=0; j<=topology[i]; j++){
            layers.back().push_back(Neuron(numOutputs, j));
            cout << "Created a Neuron!" << endl;
        }
    }
}

void NeuralNet::feedForward(const vector<double> &inputValues){
    assert(inputValues.size() == layers[0].size()-1);
    
    for (int i=0; i<inputValues.size(); ++i){
        layers[0][i].setOutputVal(inputValues[i]);
    }
    
    //Fwd Propagation
    for(int layerNum = 1; layerNum < layers.size(); layerNum++){
        Layer &prevLayer = layers[layerNum - 1];
        for (int n = 0; n< layers[layerNum].size() - 1; n++){
            layers[layerNum][n].feedForward(prevLayer);
        }
    }
}




void NeuralNet::backProp(const vector<double> &targetVals){
    // Calculate error (RMS)
    Layer &outputLayer = layers.back();
    error = 0;
    
    for (unsigned i = 0; i<outputLayer.size() - 1; i++){
        double delta = targetVals[i] - outputLayer[i].getOutputVal();
        error += delta*delta;
    }
    
    error /= outputLayer.size() - 1;
    error = sqrt(error);
    
    averageError = (averageError*smoothing + error)/(smoothing + 1.0);
    
    // Output gradients
    
    for (unsigned i = 0; i < outputLayer.size() - 1; i++){
        outputLayer[i].calcOutputGradients(targetVals[i]);
    }
    
    //Calculate gradients of hidden
    
    for(int layerNum = (int) layers.size() - 2; layerNum > 0; layerNum--){
        Layer &hiddenLayer = layers[layerNum];
        Layer &nextLayer = layers[layerNum + 1];
        
        for(int j = 0; j<hiddenLayer.size(); j++){
            hiddenLayer[j].calcHiddenGradients(nextLayer);
        }
    }

    
    //Update connection rates
    for(int layerNum = (int) layers.size() -1; layerNum > 0; layerNum--){
        Layer &layer = layers[layerNum];
        Layer &prevLayer = layers[layerNum - 1];
        
        for(int j=0; j<layer.size(); j++){
            layer[j].updateInputWeights(prevLayer);
        }
    }
    
}

void NeuralNet::getResults(vector<double> &resultsVals) const{
    resultsVals.clear();
    
    for(int i=0; i<layers.back().size(); i++)
        resultsVals.push_back(layers.back()[i].getOutputVal());
    
    
}
