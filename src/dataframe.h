#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include"wmath.h"

using namespace std;
using Eigen::MatrixXcd;
using namespace wmath;
class dataframe {

public:
int rows;
int cols;
MatrixXcd data; 
string description;

dataframe();
void load(vector<string> fnames);
void trim();

};