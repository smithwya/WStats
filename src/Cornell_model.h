#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Cornell_model : public WModel
{
public:
    Cornell_model() : WModel(){};
    Cornell_model(Eigen::VectorXd ind, Eigen::VectorXd s, string d) : WModel(3, ind, s, d){};

    double evaluate_pt(double *x, const double *pars)
    {

        return (pars[0] + pow(pars[1], 2) / (x[0]) + pars[2] * x[0]);
    };

    double extract_observable(const double *pars)
    {
        return pars[2];
    }

    double extract_error(const double *pars, const double *errs)
    {
        return errs[2];
    }

    friend ostream &operator<<(std::ostream &os, Cornell_model const &m)
    {

        os << "Fitting Cornell with shape " << m.data_shape.transpose() << endl;
        return os;
    }
};