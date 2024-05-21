#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Exp_model : public WModel
{

    Eigen::VectorXd evaluate(const double *xx)
    {
        int poly_degree = hyperparams[0];
        int pole_degree = hyperparams[1];
        int t_start = hyperparams[2];
        int t_end = hyperparams[3];
        int t_length = hyperparams[4];
        int num_params = poly_degree + pole_degree;
        Eigen::VectorXd result = Eigen::VectorXd::Zero(t_length);

        for (int T = t_start; T <= t_end; T++)
        {

            double exponent = 0;
            for (int i = 0; i <= poly_degree; i++)
            {

                exponent += xx[i] * pow(T + 1, i);
            }

            double pole_term = 1;

            for (int j = 0; j < pole_degree; j++)
            {
                pole_term += xx[poly_degree + j] * pow(T + 1, -j - 1);
            }

            result(T) = (TMath::Exp(-exponent)) * pole_term;
        }

        return result;
    };

    friend ostream &operator<<(std::ostream &os, Exp_model const &m)
    {
        int poly_degree = m.hyperparams[0];
        int pole_degree = m.hyperparams[1];
        int t_start = m.hyperparams[2];
        int t_end = m.hyperparams[3];
        int t_length = m.hyperparams[4];
        int num_params = poly_degree + pole_degree;

        os<<"(* Fitting from t_start = "<<t_start+1<<" to t_end = "<<t_end+1<<" *)"<<endl;
        os<<"G[T_,xx_]:=Exp[0";
        for (int i = 1; i <= poly_degree+1; i++)
        {

            os<<"+xx[["<<i<<"]]*T^"<<i;
        }
        os<<"]*(1";

        for (int j = 1; j < pole_degree+1; j++)
        {
            os<<"+xx[["<<poly_degree + j+1<<"]]/(T^"<<j<<")";
        }
        os<<")";
        return os;
    }
};