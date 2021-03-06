// Prediction_naive_bayes.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include "classifier.h"

using namespace std;

vector <vector <double>> Load_State_CSV(string filename)
{
	ifstream in_state_(filename.c_str(), ifstream::in); //c_str()生成一个const char *指针，指向字符串的首地址
	vector<vector<double>> state_out;
	string line;
	int counter = 0;
	while (getline(in_state_, line))   // 头文件 string，第三个参数停止符,默认为 ‘\n’
	{
		istringstream iss(line);

		vector <double> x_coord;
		double state;

		while (iss >> state)   
		{
			x_coord.push_back(state);
			if (iss.peek() == ',')   // 文本使用“，”分割 
				iss.ignore();
		}
		state_out.push_back(x_coord);

		// AA - debug only
		counter++;
		//cout << "counter" << counter << endl;
		if (counter == 1) {
			for (auto k : x_coord) {
				cout << k << ", ";
			}
			cout << endl;
		}

	}
	return state_out;

}

vector<string> Load_Label(string file_name)
{
	ifstream in_label_(file_name.c_str(), ifstream::in);
	vector< string > label_out;
	string line;
	while (getline(in_label_, line))
	{
		istringstream iss(line);
		string label;
		iss >> label;
		label_out.push_back(label);
	}
	return label_out;
}

int main()
{
	vector< vector<double> > X_train = Load_State_CSV("./train_states.txt");
	vector< vector<double> > X_test = Load_State_CSV("./test_states.txt");

	////play with some features
	//int lane_width = 4;
	//for (int i = 0; i < X_train.size(); i++)
	//{
	//	X_train[i][2] = fmod(X_train[i][2], lane_width);  // 改成相对位置
	//	X_train[i].erase(X_train[i].begin());             // 去掉s的位置
	//}
	//for (int i = 0; i < X_test.size(); i++)
	//{
	//	X_test[i][2] = fmod(X_test[i][2], lane_width);
	//	X_test[i].erase(X_test[i].begin());
	//}
	
	vector< string > Y_train = Load_Label("./train_labels.txt");
	vector< string > Y_test = Load_Label("./test_labels.txt");

	cout << "X_train number of elements " << X_train.size() << endl;
	cout << "X_train element size " << X_train[0].size() << endl;
	cout << "Y_train number of elements " << Y_train.size() << endl;
	// AA - 
	cout << "X_train Front element: ";
	for (auto kk : X_train.front()) {

		cout << kk << ", ";
	}
	cout << endl;
	// - AA

	GNB gnb = GNB();  // create a Gaussian NB 

	gnb.train(X_train, Y_train);

	cout << "X_test number of elements " << X_test.size() << endl;
	cout << "X_test element size " << X_test[0].size() << endl;
	cout << "Y_test number of elements " << Y_test.size() << endl;


	int score = 0;
	for (int i = 0; i < X_test.size(); i++)
	{
		vector<double> coords = X_test[i];
		string predicted = gnb.predict(coords);
		if (predicted.compare(Y_test[i]) == 0)
		{
			score += 1;
		}
	}

	float fraction_correct = float(score) / Y_test.size();
	cout << "You got " << (100 * fraction_correct) << " % correct" << endl;

	system("pause");
    return 0;
}

