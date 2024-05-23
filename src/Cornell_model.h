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
        int R_max = shape.size();

        Eigen::VectorXd result = Eigen::VectorXd::Zero(R_max);

        for (int R = 0; R < R_max; R++)
        {
            result(R) = (xx[0]+pow(xx[1],2)/(R+1)+xx[2]*(R+1))*shape(R);
        }

        return result;
    };
};