#include <vector>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class nExp_model : public WModel
{
public:
    int n_exponentials;
    int pole_degree;

    nExp_model() : WModel()
    {
        n_exponentials = 0;
        pole_degree = 0;
    };

    nExp_model(int n_exp, int p, int n, Eigen::VectorXd ind, Eigen::VectorXd s, string d) : WModel(n, ind, s, d)
    {
        n_exponentials = n_exp;
        pole_degree = p;
        if (2*n_exponentials + p != n)
        {
            cout << "WARNING: wrong number of parameters in model" << endl;
            n_exponentials = 0;
            pole_degree = 0;
        }
    };



    double evaluate_pt( double *x, const double *pars)
    {
        double exp = 0;
        int param_index = 0;

        for(int i = 0; i <2*n_exponentials; i+=2){
            exp+= pars[i]*TMath::Exp(-pow(pars[i+1],2)*x[0]);
        }

        double pole = 1;
        int pole_pow = 1;
        for(int i = 2*n_exponentials; i < num_params; i++){
            pole+=pars[i]*pow(x[0],-pole_pow);
            pole_pow++;
        }

        return exp*pole;
    };

    friend ostream &operator<<(std::ostream &os, nExp_model const &m)
    {
        os <<"Fitting "<< m.n_exponentials<<" exponentials with a pole of degree " << m.pole_degree<<endl;
        os << "shape: " << m.data_shape.transpose();
        return os;
    }
};