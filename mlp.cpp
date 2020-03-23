
#include "mlp.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

Neuron::Neuron(int connections) {

	weight.resize(connections);
	input.resize(connections);
	delta.resize(connections);
	oldDelta.resize(connections);

	for (int i = 0; i < connections; i++) {
		weight[i] = (float)rand() / (float)RAND_MAX;
	}
	bias = (float)rand() / (float)RAND_MAX;
}
Neuron::~Neuron() {}
//*******************************************************************************

Layer::Layer(int nNeuron, int nConnections) {

	for (int i = 0; i < nNeuron; i++) {
		neuron.push_back(Neuron(nConnections));
	}

}
Layer::~Layer() {
}
//*******************************************************************************

MLP::MLP(int input, int output, vector<int> hidden, float learningRate, float momentum) {

	this->learningRate = learningRate;
	this->momentum = momentum;

	layer.push_back(Layer(hidden[0], input));

	for (int i = 1; i < hidden.size(); i++)
		layer.push_back(Layer(hidden[i], hidden[i - 1]));

	layer.push_back(Layer(output, hidden[hidden.size() - 1]));
}
MLP::~MLP() {
}
//*******************************************************************************

void MLP::training(vector<vector<float> > inputT, vector<vector<float> > outputT, int epoches) {

	float errorT, sumT, errorAntV;
	float errorUp = 0;

	for (int p = 0; p < epoches; p++) {

		sumT = 0;
		for (int i = 0; i < inputT.size(); i++) {

			for (int j = 0; j < layer[0].neuron.size(); j++)
				for (int k = 0; k < layer[0].neuron[j].input.size(); k++)
					layer[0].neuron[j].input[k] = inputT[i][k];

			feedForward(i, outputT);
			errorT = layer[layer.size() - 1].neuron[0].error;
			sumT += pow(errorT, 2);
			feedBack(i, outputT);
		}
		errorT = sumT / inputT.size();

		cout << "epoches: " << p << " Error training: " << errorT << endl;
		printStructure();
	}
}
//*******************************************************************************

void MLP::test(vector<vector<float> > input, vector<vector<float> >& output) {

	output.resize(input.size());
	for (int i = 0; i < output.size(); i++)
		output[i].resize(layer[layer.size() - 1].neuron.size());

	for (int i = 0; i < input.size(); i++) {

		for (int j = 0; j < layer[0].neuron.size(); j++)
			for (int k = 0; k < layer[0].neuron[j].input.size(); k++) {
				layer[0].neuron[j].input[k] = input[i][k];
			}
		feedForward(i, output);
		//output[i].resize(layer[layer.size()-1].neuron.size());
		for (int j = 0; j < layer[layer.size() - 1].neuron.size(); j++) {
			output[i][j] = layer[layer.size() - 1].neuron[j].output;
		}
	}
}

//*******************************************************************************
void MLP::feedForward(int input, vector<vector<float> > output) {
	// tinh lop hidden dau tien (lop hidden 1)
	// tinh output
	for (int j = 0; j < layer[0].neuron.size(); j++) {
		layer[0].neuron[j].vj = 0;
		// tinh tong input * weight
		for (int k = 0; k < layer[0].neuron[j].weight.size(); k++) {
			layer[0].neuron[j].vj += layer[0].neuron[j].weight[k] * layer[0].neuron[j].input[k];// = gia tri input[i][k];
		}
		// lay tong + bias
		layer[0].neuron[j].vj += layer[0].neuron[j].bias;
		// output = signmoidal (tong + bias)
		layer[0].neuron[j].output = sigmoidal(layer[0].neuron[j].vj);

	}

	//tinh cac lop khac
	for (int l = 1; l < layer.size(); l++) {
		for (int j = 0; j < layer[l].neuron.size(); j++) {
			layer[l].neuron[j].vj = 0;
			for (int k = 0; k < layer[l].neuron[j].weight.size(); k++) {
				layer[l].neuron[j].vj += layer[l].neuron[j].weight[k] * layer[l - 1].neuron[k].output;
			}
			layer[l].neuron[j].vj += layer[l].neuron[j].bias;
			layer[l].neuron[j].output = sigmoidal(layer[l].neuron[j].vj);
		}

	}
	//tinh sai so o lop output
	int lastLayer = layer.size() - 1;
	for (int j = layer[lastLayer].neuron.size() - 1; j >= 0; j--) {
		layer[lastLayer].neuron[j].error = output[input][j] - layer[lastLayer].neuron[j].output;

		//tinh gradient
		layer[lastLayer].neuron[j].gradient = layer[lastLayer].neuron[j].error * derivateSigmoidal(layer[lastLayer].neuron[j].output);
	}

	//tinh sai so o cac lop hidden
	for (int i = lastLayer - 1; i >= 0; i--) {
		for (int j = layer[i].neuron.size() - 1; j >= 0; j--) {
			layer[i].neuron[j].error = layer[i].neuron[j].output * (1 - layer[i].neuron[j].output) * layer[lastLayer].neuron[0].output * layer[i].neuron[0].weight[j];

			//tinh gradient o cac lop hidden
			float s = 0;
			for (int k = 0; k < (int)layer[i + 1].neuron.size(); k++) {
				s += layer[i + 1].neuron[k].weight[j] * layer[i + 1].neuron[k].gradient;
			}
			layer[i].neuron[j].gradient = s * derivateSigmoidal(layer[i].neuron[j].output);
		}
	}
}

//*******************************************************************************
void MLP::feedBack(int input, vector<vector<float> > output) {

	int lastLayer = layer.size() - 1;
	//cap nhat weights va bias
	for (int i = lastLayer; i >= 1; i--) {
		for (int j = 0; j < layer[i].neuron.size(); j++) {
			for (int k = 0; k < layer[i].neuron[j].weight.size(); k++) {

				layer[i].neuron[j].delta[k] = learningRate * (layer[i].neuron[j].gradient * layer[i - 1].neuron[j].output);
				layer[i].neuron[j].weight[k] += layer[i].neuron[j].delta[k] + (layer[i].neuron[j].oldDelta[k] * momentum);
				layer[i].neuron[j].oldDelta[k] = layer[i].neuron[j].delta[k];

			}
			layer[i].neuron[j].bias += (learningRate * layer[i].neuron[j].gradient);// * layer[i].neuron[j].error;
		}
	}
	//cap nhat weight va bias dau vao
	for (int j = 0; j < layer[0].neuron.size(); j++) {
		for (int k = 0; k < layer[0].neuron[j].weight.size(); k++) {

			layer[0].neuron[j].delta[k] = learningRate * (layer[0].neuron[j].gradient * layer[0].neuron[j].input[k]);
			layer[0].neuron[j].weight[k] += layer[0].neuron[j].delta[k] + (layer[0].neuron[j].oldDelta[k] * momentum);
			layer[0].neuron[j].oldDelta[k] = layer[0].neuron[j].delta[k];

		}
		layer[0].neuron[j].bias += (learningRate * layer[0].neuron[j].gradient);
	}
}

//*******************************************************************************
float MLP::sigmoidal(float vj) {
	return 0.5 * ((vj / (abs(vj) + 1)) + 1);
}

float MLP::derivateSigmoidal(float vj) {
	return vj * (1 - vj);
}

//*******************************************************************************
float MLP::radial(float vj) {
	return 1 / (1 + (-vj));
}

//*******************************************************************************
void MLP::printStructure() {

	cout << "\t\n Structure \n" << endl;
	cout << "Input: " << layer[0].neuron[0].weight.size() << endl;
	cout << "Hidden layers: " << layer.size() - 1 << endl;
	for (int i = 0; i < layer.size() - 1; i++)
		cout << "Hidden [" << i << "] " << " neuron: " << layer[i].neuron.size() << endl;
	cout << "Output: " << layer[layer.size() - 1].neuron.size() << endl << endl;


	for (int i = 0; i < layer.size(); i++) {
		cout << "Hidden: " << i << " neuron: " << layer[i].neuron.size() << endl;
		for (int j = 0; j < layer[i].neuron.size(); j++) {
			cout << " Neuron: " << j << endl;
			for (int k = 0; k < layer[i].neuron[j].weight.size(); k++) {
				cout << "  Weight [" << k << "]: " << layer[i].neuron[j].weight[k] << endl;
			}
			cout << "  Bias: " << layer[i].neuron[j].bias << endl;
			cout << "  Sum of input (W * a): " << layer[i].neuron[j].vj << endl;
			cout << "  Output (a): " << layer[i].neuron[j].output << endl;
			cout << "  Error: " << layer[i].neuron[j].error << endl << endl;
		}

	}

}


//*******************************************************************************
