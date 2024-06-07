#include <vector>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
#include "TROOT.h"
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


    double extract_observable(const double* pars){

        double num = 0;
        double denom = 0;
        for(int i = 0; i <2*n_exponentials; i+=2){
        num+= pow(pars[i],2)*pow(pars[i+1],2);
        denom+=pow(pars[i],2);
        }

        return num/denom;
    }

    
    double extract_error(const double* pars, const double* errs){
        double geo_sum = 0;
        double geo_err=0;
        double amp_sum=0;
        double amp_err = 0;

        double result = 0;


        for(int i = 0; i <2*n_exponentials; i+=2){

            double a = pow(pars[i],2);
            double da = 2*sqrt(a)*errs[i];

            double m = pow(pars[i+1],2);
            double dm = 2*sqrt(m)*errs[i+1];
            
            amp_sum+=a;
            geo_sum+=a*m;

            amp_err+=pow(da,2);
            geo_err+=pow(a*m*sqrt(pow(da/a,2)+pow(dm/m,2)),2);
        }
        amp_err = sqrt(amp_err);
        geo_err = sqrt(geo_err);
        result = sqrt(pow(amp_err/amp_sum,2)+pow(geo_err/geo_sum,2));
        
        result = result * (geo_sum/amp_sum);
        return result;
    }

    double evaluate_pt( double *x, const double *pars)
    {
        double exp = 0;
        int param_index = 0;

        for(int i = 0; i <2*n_exponentials; i+=2){
            exp+= pow(pars[i],2)*TMath::Exp(-pow(pars[i+1],2)*x[0]);
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
        os <<m.n_exponentials<<" exponentials with a pole of degree " << m.pole_degree<<endl;
        os << "shape: " << m.data_shape.transpose();
        return os;
    }
};