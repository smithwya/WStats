#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Polynomial: public WModel
{
    public:

    Polynomial() : WModel(){};

    Polynomial(int n,Eigen::VectorXd ind, Eigen::VectorXd s, string d) : WModel(n, ind, s, d){};

    double evaluate_pt( double *x, const double *pars)
    {
        double result = 0;

        for(int i = 0; i < num_params; i++){
            result+=pars[i]*pow(x[0],i);

        }
        return result;
    };


    double extract_observable(const double* pars){
        return pars[1];
    }


    double extract_error(const double* pars, const double* errs){
        return errs[1];
    }

        friend ostream &operator<<(std::ostream &os, Polynomial const &m)
    {

        os << "Fitting Polynomial of degree" <<m.num_params-1<<" with shape "<< m.data_shape.transpose() << endl;
        return os;
    }
};