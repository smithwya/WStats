#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include"wmath.h"

using namespace std;

namespace wmath{


    //Evaluate a function over every column of a matrix
    Eigen::MatrixXcd eval_function(const Eigen::MatrixXcd &data, Eigen::VectorXcd (*func)(Eigen::VectorXcd)){
        const int n_variates=data.rows();
        const int n_samples=data.cols();
        const int dim_output = func(data.col(0)).rows();
        if(n_samples<=0) return Eigen::MatrixXcd::Zero(1,1);

        Eigen::MatrixXcd evaluated_func = Eigen::MatrixXcd::Zero(dim_output,n_samples);

        for(int i = 0; i < n_samples; i++){
            evaluated_func.col(i)=func(data.col(i));
        }

        return evaluated_func;
    }

/*
    //Given a matrix of values, each col is a separate set of values, give a matrix with the jackknifed result in each column
    Eigen::MatrixXcd jackknife_sample(const Eigen::MatrixXcd &data){
        const int n_vals = data.rows();
        const int n_samples = data.cols();
        Eigen::MatrixXcd jack_samples = Eigen::MatrixXcd::Zero(n_vals,n_samples);
        Eigen::VectorXcd avg=data.colwise().mean();

        for(int i = 0; i < n_samples; i++){

            jack_samples.col(i) = (n_samples*avg-data.col(i))/(n_samples-1);

        }
        return jack_samples;
    }

    vector<Eigen::VectorXcd> jackknife_average(const Eigen::MatrixXcd &data){
        const int n_samples = data.cols();
        const int n_variates = data.rows();

        Eigen::MatrixXcd jack_samples = jackknife_sample(data);
        Eigen::VectorXcd avg = data.colwise().mean();
        Eigen::VectorXcd jack_average = jack_samples.colwise().mean();
        Eigen::VectorXcd jack_variance = Eigen::VectorXcd::Zero(n_variates);

        for(int i = 0; i < n_samples; i++){
            jack_variance += (jack_samples.col(i)-avg).cwiseAbs2();
        }

        return {jack_average,jack_variance*(n_samples-1)/n_samples};


    }


    Eigen::MatrixXcd sample_covariance(const Eigen::MatrixXcd &data){
        const int n_samples = data.cols();
        Eigen::MatrixXcd centered_dat = data.colwise()-data.colwise().mean();
        Eigen::MatrixXcd cov = centered_dat.adjoint()*centered_dat/(n_samples-1);
        return cov;
    }
    */
}