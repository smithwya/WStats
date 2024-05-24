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

    Polynomial(int n, Eigen::VectorXd s, string d) : WModel(n, s, d){};

    Eigen::VectorXd evaluate(const double *xx)
    {
        int R_max = shape.size();
        Eigen::VectorXd result = Eigen::VectorXd::Zero(R_max);

        for (int R = 0; R < R_max; R++)
        {
            double temp = 0;

            for(int i = 0; i < num_params; i++){
                temp+=xx[i]*pow(R+1,i);

            }

            result(R) = temp*shape(R);
        }

        return result;
    };
};