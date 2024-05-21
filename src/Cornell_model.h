#include <vector>
#include <string>
#include <Eigen/Dense>
#include "TMath.h"
#include "WModel.h"
using namespace std;

class Cornell_model: public WModel
{

    Eigen::VectorXd evaluate(const double *xx)
    {
        int r_start = hyperparams[0];
        int r_end = hyperparams[1];
        int r_length = hyperparams[2];

        Eigen::VectorXd result = Eigen::VectorXd::Zero(r_length);

        for (int R = r_start; R <= r_end; R++)
        {
            result(R) = xx[0]+pow(xx[1],2)/(R+1)+xx[2]*(R+1);
        }

        return result;
    };
};