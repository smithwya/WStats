#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include "WMath.h"
#include <TRandom3.h>

using namespace std;

namespace WMath
{

    // Evaluate a function over every column of a matrix
    Eigen::MatrixXd eval_function(const Eigen::MatrixXd &data, Eigen::VectorXd (*func)(Eigen::VectorXd))
    {
        const int n_variates = data.rows();
        const int n_samples = data.cols();
        const int dim_output = func(data.col(0)).rows();
        if (n_samples <= 0)
            return Eigen::MatrixXd::Zero(1, 1);

        Eigen::MatrixXd evaluated_func = Eigen::MatrixXd::Zero(dim_output, n_samples);

        for (int i = 0; i < n_samples; i++)
        {
            evaluated_func.col(i) = func(data.col(i));
        }

        return evaluated_func;
    };

    // Given a matrix of values, each col is a separate set of values, give a matrix with the jackknifed result in each column
    Eigen::MatrixXd jackknife_sample(const Eigen::MatrixXd &data)
    {
        const int n_vals = data.rows();
        const int n_samples = data.cols();
        
        Eigen::MatrixXd jack_samples = Eigen::MatrixXd::Zero(n_vals, n_samples);
        Eigen::VectorXd avg = data.rowwise().mean();

        for (int i = 0; i < n_samples; i++)
        {

            jack_samples.col(i) = (n_samples * avg - data.col(i)) / (n_samples - 1);
        }
        return jack_samples;
    };

    vector<Eigen::VectorXd> jackknife_average(const Eigen::MatrixXd &data)
    {
        const int n_samples = data.cols();
        const int n_variates = data.rows();

        Eigen::MatrixXd jack_samples = jackknife_sample(data);
        Eigen::VectorXd avg = data.colwise().mean();
        Eigen::VectorXd jack_average = jack_samples.colwise().mean();
        Eigen::VectorXd jack_variance = Eigen::VectorXd::Zero(n_variates);

        for (int i = 0; i < n_samples; i++)
        {
            jack_variance += (jack_samples.col(i) - avg).cwiseAbs2();
        }

        return {jack_average, jack_variance * (n_samples - 1) / n_samples};
    };



    Eigen::VectorXd gen_shape(int length, int start_index, int stop_index)
    {
        if (stop_index < start_index || stop_index > length - 1 || start_index < 0)
            return Eigen::VectorXd::Zero(1);

        Eigen::VectorXd shape = Eigen::VectorXd::Zero(length);
        for (int i = start_index; i <= stop_index; i++)
        {
            shape(i) = 1;
        }
        return shape;
    }


}