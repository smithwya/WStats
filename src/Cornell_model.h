#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Cornell_model: public WModel
{
    public:


    double evaluate_pt(const double *pars, double x)
    {
       
        return (pars[0]+pars[1]/(x)+pars[2]*x);
    };
};