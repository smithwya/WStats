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
    Eigen::VectorXd data_shape;//length is the length of data
    Eigen::VectorXd ind_vars;//x values for each entry in a sample
    vector<int> used_indices;//calculated based off the shape, the list of indices in ind_vars which are used


    Eigen::VectorXd evaluate(const double *xx)
    {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(used_indices.size());

        for(int i = 0; i < used_indices.size(); i++){
            result(i) = evaluate_pt(&ind_vars(used_indices[i]), xx);
        }
        return result;
    };

    virtual double evaluate_pt(double *x,const double *pars){
        return 0;
    }

    WModel(int n, Eigen::VectorXd ind, Eigen::VectorXd s, string d)
    {
        num_params = n;
        description = d;
        data_shape = s;
        ind_vars = ind;
        used_indices = {};

        if(s.size() != ind.size()){ cout<<"WARNING: invalid model"<<endl;
        return;
        }

        for(int i = 0; i <ind_vars.size(); i++){
            if(data_shape[i]==1) used_indices.push_back(i);
        }

    }

    WModel()
    {
        description = "";
        num_params = 0;
        data_shape = Eigen::VectorXd::Zero(1);
        ind_vars = Eigen::VectorXd::Zero(1);
        used_indices = {};
    };

    friend ostream &operator<<(std::ostream &os, WModel const &m)
    {
        os << m.description << endl;
        os << "evaluated at: ";
        for(int x : m.used_indices) os <<m.ind_vars[x]<<" ";
        os<<endl;
        os << "number of params: " << m.num_params << endl;
        os << "shape: "<<endl;
        os << m.data_shape;
        return os;
    }
};