#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include"wmath.h"


using namespace std;
using Eigen::MatrixXcd;
using namespace wmath;
class model {
public:
vector<complex<double>> params;
string description;
complex<double> evaluate(vector<complex<double>> indep_vars);

};