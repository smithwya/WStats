#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Cornell_model: public WModel
{
    public:
    Cornell_model() : WModel(){};
    Cornell_model(Eigen::VectorXd ind, Eigen::VectorXd s, string d) : WModel(3,ind,s,d){};

    double evaluate_pt(double *x,const double *pars)
    {
       
        return (pars[0]+pars[1]/(x[0])+pars[2]*x[0]);
    };

    friend ostream &operator<<(std::ostream &os, Cornell_model const &m)
    {

        os << "(* Fitting with shape " << m.data_shape.transpose() << " *)" << endl;
        os << "G[R_,xx_]:=xx[[0]] + xx[[1]]/R | xx[[2]]*R";
        return os;
    }
};