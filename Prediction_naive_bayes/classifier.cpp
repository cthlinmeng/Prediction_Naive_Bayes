#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include "classifier.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926
#endif // !M_PI


using namespace std;
/**
* Initializes GNB
*/
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double> > data, vector<string> labels)
{
	//计算每个label下各个独立变量的mean与std，每个变量满足gaussian分布，后续用来计算概率
	/*
	Trains the classifier with N data points and labels.

	INPUTS
	data - array of N observations
	- Each observation is a tuple with 4 values: s, d,
	s_dot and d_dot.
	- Example : [
	[3.5, 0.1, 5.9, -0.02],
	[8.0, -0.3, 3.0, 2.2],
	...
	]

	labels - array of N labels
	- Each label is one of "left", "keep", or "right".
	*/

	cout << "---- Training ---- " << endl;

	int features_count = data[0].size();
	cout << "Train: Features count:" << features_count << endl;

	unique_labels_ = labels;
	// find # of unique labels
	vector<string>::iterator sit;
	std::sort(unique_labels_.begin(), unique_labels_.end());   // keep left right 首字母排序
	sit = std::unique(unique_labels_.begin(), unique_labels_.end());  // 返回去重后最后一个元素的地址
	unique_labels_.erase(sit, unique_labels_.end());  // remove duplicate 只剩下3个 

	map<string, vector<vector<double>> > lfkData;  // map of Label --> Table of vector of features

	// 0 - Inits
	for (auto label : unique_labels_) {
		class_counts_[label] = 0; // init class count
		vector<double> tmp(features_count, 0.0);  // vector of four 0.0
		means_.insert(std::pair<string, vector<double> >(label, tmp));
		stds_.insert(std::pair<string, vector<double> >(label, tmp));
	}

	int TRAIN_SIZE = labels.size();   // size of X or Y
									  // 1 - collect data info: class counts; means, vars
	for (auto i = 0; i < TRAIN_SIZE; i++) {
		// 1a - class count
		class_counts_[labels[i]] += 1;
		// 1b - store X_train per label (or class)
		lfkData[labels[i]].push_back(data[i]);
		// 1c - Sum for Mean calculation
		for (auto j = 0; j < features_count; j++) {
			means_[labels[i]][j] += data[i][j];  // SUM - for mean
		}
	}

	// 2 - calc Mean for each Feature
	for (auto label : unique_labels_) {
		p_class_[label] = class_counts_[label] * 1.0 / labels.size(); // prob of label (or class)

		for (auto j = 0; j < features_count; j++) {   // for each feature
			means_[label][j] /= class_counts_[label];  // MEAN
		}
	}

	// 3 - calc Variance: 
	// ...for each Label
	for (auto label : unique_labels_) {
		// ...for each Feature
		for (auto j = 0; j < features_count; j++) {
			vector<double> diff_sq(4, 0.0);
			// ...for each Row
			for (auto row = 0; row < lfkData[label].size(); row++) {
				// accumulate Squared Difference
				stds_[label][j] += pow(lfkData[label][row][j] - means_[label][j], 2.0);
			}
			stds_[label][j] /= class_counts_[label];  // calc Var
			stds_[label][j] = sqrt(stds_[label][j]);  // calc Sigma
		}
	}

	// Debug
	cout << "Unique labels: " << endl;
	for (auto label : unique_labels_) {     // should be just KEEP, LEFT, RIGHT
		cout << label << " has samples: " << class_counts_[label] << endl;

		cout << " Mean: "; //
		for (const auto &j : means_[label]) {
			cout << j << ", ";
		}
		cout << endl;

		cout << " Std:  "; //
		for (const auto &k : stds_[label]) {
			cout << k << ", ";
		}
		cout << endl;

		cout << " Prob: ";
		cout << p_class_[label] << endl;

	}

	cout << "--- End Training ---" << endl;
}

string GNB::predict(vector<double> vec)
{
	/*
	Once trained, this method is called and expected to return
	a predicted behavior for the given observation.

	INPUTS

	observation - a 4 tuple with s, d, s_dot, d_dot.
	- Example: [3.5, 0.1, 8.5, -0.2]

	OUTPUT

	A label representing the best guess of the classifier. Can
	be one of "left", "keep" or "right".

	NOTES:
	-----
	Assuming f1, f2, f3, f4 are each Features (assumed INDEPENDENT - hence *Naive* Bayes)

	P(LabelA|f1, f2, f3, f4) = P(LabelA) * P(f1|LabelA) * P(f2|LabelA) * P(f3|LabelA) * P(f4|LabelA)
	P(LabelB|f1, f2, f3, f4) = P(LabelB) * P(f1|LabelB) * P(f2|LabelB) * P(f3|LabelB) * P(f4|LabelB)
	P(LabelC|f1, f2, f3, f4) = P(LabelC) * P(f1|LabelC) * P(f2|LabelC) * P(f3|LabelC) * P(f4|LabelC)

	Then return the highest probability --> that is the Label

	Calculating Gaussian Probability:

	P(f1|LabelA) = norm * exp(-num / denom),
	where num = (obs_f1 - mean_f1)^2      // obs for feature f1
	norm = 1.0 / sqrt ( 2 * M_PI * std_dev_f1^2)
	denom = 2 * std_dev_f1^2

	"""
	# TODO - complete this
	*/

	map<string, double> p;
	double max = 0;
	string result;

	for (auto label : unique_labels_) {
		p[label] = p_class_[label];   // prior prob of Label

		for (auto j = 0; j < vec.size(); j++) { // for each feature (column) of input vector
			double norm = 1.0 / sqrt(2 * M_PI * pow(stds_[label][j], 2));
			double num = pow(vec[j] - means_[label][j], 2);
			double denom = 2 * pow(stds_[label][j], 2);
			p[label] *= norm * exp(-num / denom);
		}
		if (max < p[label]) {
			max = p[label];
			result = label;
		}
	}

	return result;

}