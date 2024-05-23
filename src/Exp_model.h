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

    Exp_model(int e, int p, int n, Eigen::VectorXd s, string d) : WModel(n, s, d)
    {
        exp_degree = e;
        pole_degree = p;
        if (e + p + 1 != n)
        {
            cout << "Warning: invalid model" << endl;
            exp_degree = 0;
            pole_degree = 0;
        }
    };

    Eigen::VectorXd evaluate(const double *xx)
    {
        int T_max = shape.size();

        Eigen::VectorXd result = Eigen::VectorXd::Zero(T_max);

        for (int T = 0; T < T_max; T++)
        {

            double exponent = 0;
            for (int i = 0; i <= exp_degree; i++)
            {

                exponent += xx[i] * pow(T + 1, i);
            }

            double pole_term = 1;

            for (int j = 0; j < pole_degree; j++)
            {
                pole_term += xx[exp_degree + j] * pow(T + 1, -j - 1);
            }

            result(T) = (TMath::Exp(-exponent)) * pole_term * shape(T);
        }
        return result;
    };

    friend ostream &operator<<(std::ostream &os, Exp_model const &m)
    {

        os << "(* Fitting with shape " << m.shape.transpose() << " *)" << endl;
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