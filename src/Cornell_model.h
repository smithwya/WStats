#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Cornell_model: public WModel
{
    public:


    double evaluate_pt(double *x,const double *pars)
    {
       
        return (pars[0]+pars[1]/(x[0])+pars[2]*x[0]);
    };
};