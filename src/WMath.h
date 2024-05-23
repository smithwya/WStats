#include <vector>
#include <complex>
#include <Eigen/Dense>

using namespace std;

namespace WMath
{

    // Evaluate a function over every column of a matrix
    Eigen::MatrixXd eval_function(const Eigen::MatrixXd &data, Eigen::VectorXd (*func)(Eigen::VectorXd));
    // Given a matrix of values, each col is a separate set of values, give a matrix with the jackknifed result in each column
    Eigen::MatrixXd jackknife_sample(const Eigen::MatrixXd &data);
    vector<Eigen::VectorXd> jackknife_average(const Eigen::MatrixXd &data);

    std::vector<Eigen::MatrixXd> jackknife_sample(const std::vector<Eigen::MatrixXd>);
    Eigen::VectorXd gen_shape(int length, int start_index, int stop_index);
}