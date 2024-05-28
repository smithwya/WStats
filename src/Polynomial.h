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

    double evaluate_pt(const double *pars, double x)
    {
        double result = 0;

        for(int i = 0; i < num_params; i++){
            result+=pars[i]*pow(x,i);

        }
        return result;
    };
};