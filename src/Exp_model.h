#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Exp_model : public WModel
{
public:
    int exp_degree;
    int pole_degree;

    Exp_model() : WModel()
    {
        exp_degree = 0;
        pole_degree = 0;
    };

    Exp_model(int e, int p, int n, Eigen::VectorXd ind, Eigen::VectorXd s, string d) : WModel(n, ind, s, d)
    {
        exp_degree = e;
        pole_degree = p;
        if (e + p + 1 != n)
        {
            cout << "WARNING: wrong number of parameters in model" << endl;
            exp_degree = 0;
            pole_degree = 0;
        }
    };


    double evaluate_pt( double *x, const double *pars)
    {
        double result = 0;
        double exponent = 0;

        for (int i = 0; i <= exp_degree; i++)
        {

            exponent += pars[i] * pow(x[0], i);
        }

        double pole_term = 1;

        for (int j = 1; j <= pole_degree; j++)
        {
            pole_term += pars[exp_degree + j] * pow(x[0], -j);
        }

        return (TMath::Exp(-exponent)) * pole_term;
    };

    friend ostream &operator<<(std::ostream &os, Exp_model const &m)
    {

        os << "(* Fitting with shape " << m.data_shape.transpose() << " *)" << endl;
        os << "G[T_,xx_]:=Exp[0";
        for (int i = 1; i <= m.exp_degree + 1; i++)
        {

            os << "+xx[[" << i << "]]*T^" << i;
        }
        os << "]*(1";

        for (int j = 1; j < m.pole_degree + 1; j++)
        {
            os << "+xx[[" << m.exp_degree + j + 1 << "]]/(T^" << j << ")";
        }
        os << ")";
        return os;
    }
};