#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include "WMath.h"

using namespace std;
using Eigen::MatrixXcd;
using namespace WMath;
class WModel
{
public:
    int num_params;
    string description;
    Eigen::VectorXd evaluate(const double *xx){
        return Eigen::VectorXd::Zero(1);
    };
    WModel(){
        num_params = 0;
        description = "";
    };
};