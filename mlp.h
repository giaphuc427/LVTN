#include <stdio.h>
#include <vector>

#ifndef MLP_H
#define MLP_H

using namespace std;

class Neuron {
public:
	Neuron();
	Neuron(int connections);
	~Neuron();

	vector <float> weight, oldWeight, delta, oldDelta, input;
	float bias, oldBias, output, error;

	float vj;
	float gradient;
};
class Layer {
public:
	Layer();
	Layer(int nNeuronio, int nConnections);
	~Layer();

	vector <Neuron> neuron;
};

class MLP {
public:
	MLP();
	MLP(int input, int output, vector<int> hidden, float learningRate = 0.5, float momentum = 0.6);
	~MLP();
	void training(vector<vector<float> > inputT, vector<vector<float> > outputT, int epoches);
	void test(vector<vector<float> > input, vector<vector<float> > &output);
	void feedForward(int input, vector<vector<float> > output);
	void feedBack(int input, vector<vector<float> > output);

	float sigmoidal(float vj);
	float derivateSigmoidal(float vj);
	float radial(float vj);

	static void testMlp();
	void printStructure();

	vector <Layer> layer;
	float learningRate, momentum;
	float errorV;


};

#endif
