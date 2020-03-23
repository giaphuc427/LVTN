//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include "mlp.h"
//#include "genetic.h"

using namespace std;

void loadTXTFile(vector<vector<float> > &samples, string path) {

	ifstream file;
	file.open(path);
	if (!file.is_open()) {
		cout << "Erro ao abrir arquivo: " << path << endl;
		return;
	}

	int height, width;
	file >> height; file >> width;

	samples.resize(height);
	for (int i = 0; i < (int)samples.size(); i++) {
		samples[i].resize(width);
		for (int j = 0; j < (int)samples[i].size(); j++) {
			file >> samples[i][j];
			if (isnan(samples[i][j]))
				samples[i][j] = 0.0;
		}

	}
	file.close();

}

void normalize(vector<vector<float> >&samples) {


	vector<float>max(samples[0].size(), -11), min(samples[0].size(), 233);

	for (int j = 0; j < (int)min.size(); j++) {

		for (int i = 0; i < (int)samples.size(); i++) {
			if (max[j] == -11 || max[j] < samples[i][j])
				max[j] = samples[i][j];

			if (min[j] == 233 || min[j] > samples[i][j])
				min[j] = samples[i][j];
		}
	}

	for (int j = 0; j < (int)min.size(); j++) {

		if (min[j] > 0)
			min[j] = 0;

		if (max[j] == 0)
			max[j] = 1;

		for (int i = 0; i < (int)samples.size(); i++)
			samples[i][j] = (samples[i][j] - min[j]) / (max[j] - min[j]);

	}
}

void saveTXTFile(vector<vector<float> > samples, string path, bool save_to_c) {

	ofstream file;
	file.open(path);
	if (!file.is_open()) {
		cout << "Erro ao abrir arquivo: " << path + ".txt" << endl;
		return;
	}
	if (save_to_c == 1)
		file << samples.size() << " " << samples[0].size() << endl;

	for (int i = 0; i < (int)samples.size(); i++) {
		for (int j = 0; j < (int)samples[i].size(); j++) {

			file << samples[i][j] << " ";
		}file << endl;

	}
	file.close();
}

int main() {
	vector<vector<float> > inputTraining, outputTraining, inputValidate, outputValidate, inputTest, outputTest;
	vector<int>hidden(2);
	hidden[0] = 2;
	hidden[1] = 2;

	loadTXTFile(inputTraining, "test/Training_in1.txt");
	loadTXTFile(outputTraining, "test/Training_out1.txt");

	loadTXTFile(inputTest, "test/Testing_in.txt");

	MLP *mlp = new MLP(inputTraining[0].size(), outputTraining[0].size(), hidden, 0.9, 0.6);

	mlp->training(inputTraining, outputTraining, 10);

	mlp->test(inputTest, outputTest);

	mlp->printStructure();

	saveTXTFile(outputTest, "result/TestOutMLP", 1);

	saveTXTFile(outputTest, "result/TestOutMLP.txt", 1);
	system("pause");
	return 0;

}