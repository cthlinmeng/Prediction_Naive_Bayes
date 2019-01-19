#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

class GNB {
public:

	vector<string> classes = { "left","keep","right" };

	vector<string>  unique_labels_;       // Left, Right, Keep

	map<string, vector<double> > means_;  // Map Label --> Mean for each Feature
	map<string, vector<double> > stds_;  // Map Label --> Std Dev for each Feature
	map<string, int>  class_counts_;     // count of Label (L, K, R) datapoints
	map<string, double> p_class_;       // prob of L, K, or R


										/**
										* Constructor
										*/
	GNB();

	/**
	* Destructor
	*/
	virtual ~GNB();

	void train(vector<vector<double> > data, vector<string>  labels);

	string predict(vector<double>);

};

