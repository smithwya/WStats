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
    vector<double> hyperparams;
    string description;
    virtual Eigen::VectorXd evaluate(const double *xx) { return Eigen::VectorXd::Zero(1); };
    //   static WModel *Create(vector<double> hyper);
    WModel(vector<double> hp, int n, string d) { hyperparams = hp; num_params = n; description = d;}
    void init(vector<double> hp, int n, string d){
        hyperparams = hp; num_params = n; description = d;
    };
    WModel()
    {
        hyperparams = {};
        description = "";
        num_params = 0;
    };
    friend ostream& operator<<(std::ostream& os, WModel const& m){
        os<<m.description<<endl;
        os<<m.num_params<<endl;
        for(double d: m.hyperparams){
            os<<d<<" ";
        }
        os<<endl;
    }
};