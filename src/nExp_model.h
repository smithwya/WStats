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
        
        var_lims = Eigen::MatrixXd::Zero(n,2);
        for(int i = 0; i <2*n_exponentials; i+=2){
            var_lims(i,0)=0;
            var_lims(i,1)=5;
            var_lims(i+1,0)=0.001;
            var_lims(i+1,1)=5;
        }
        
        for(int i = 2*n_exponentials; i < n; i++){
            var_lims(i,0)=0;
            var_lims(i,1)=1;
        }
        

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
        double mass = 0;
        for(int i = 0; i <2*n_exponentials; i+=2){
            mass+=pars[i+1];
            num+= pars[i]*mass;
            denom+=pars[i];
        }
        
        //return num/denom;
        return pars[1];
    }

    
    double extract_error(const double* pars, const double* errs){

       namespace unc = uncertainties;

       unc::udouble mass(0,0);
       unc::udouble numer(0,0);
       unc::udouble denom(0,0);


        for(int i = 0; i <2*n_exponentials; i+=2){
            unc::udouble tmp_amp(pars[i],errs[i]);
            unc::udouble tmp_mass(pars[i+1],errs[i+1]);

            mass+=tmp_mass;
            numer+=tmp_amp*mass;
            denom+=tmp_amp;
        }
        
        unc::udouble result = numer/denom;

        //return result.s();
        return errs[1];
    }

    double evaluate_pt( double *x, const double *pars)
    {
        double exp = 0;
        int param_index = 0;
        double mass = 0;

        for(int i = 0; i <2*n_exponentials; i+=2){
            mass+=pars[i+1];
            exp+= pars[i]*TMath::Exp(-mass*x[0]);
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