#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
using namespace std;

class WModel
{
public:
    int num_params;
    string description;
    Eigen::VectorXd shape;

    virtual Eigen::VectorXd evaluate(const double *xx)
    {
        return shape;
    };
    
    WModel(int n, Eigen::VectorXd s, string d)
    {
        num_params = n;
        description = d;
        shape = s;
    }

    WModel()
    {
        description = "";
        num_params = 0;
        shape = Eigen::VectorXd::Zero(1);
    };



    friend ostream &operator<<(std::ostream &os, WModel const &m)
    {
        os << m.description << endl;
        os << m.num_params << endl;
        os << m.shape;
        return os;
    }
};