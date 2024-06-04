#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include "WMath.h"

using namespace std;
using namespace WMath;
class WFrame
{

public:
    int n_samples;
    Eigen::MatrixXd data;
    string description;
    Eigen::VectorXd indep_vars;
    Eigen::MatrixXd cov_matrix;

    WFrame()
    {
        n_samples = 0;
        data = Eigen::MatrixXd::Zero(1, 1);
        indep_vars = Eigen::MatrixXd::Zero(1, 1);
        description = "";
        cov_matrix = Eigen::MatrixXd::Zero(1, 1);
    };

    WFrame(Eigen::MatrixXd d)
    {
        data = d;
        n_samples = d.cols();
        indep_vars = Eigen::VectorXd::Ones(d.cols());
        for (int i = 0; i < d.cols(); i++)
        {
            indep_vars(i) = i + 1;
        }
        if (d.cols() > d.rows())
        {
            cov_matrix = WFrame::cov(data);
        }
        else
        {
            cov_matrix = Eigen::MatrixXd::Ones(1, 1);
            cout << "Warning: Not enough samples to calculate covariance matrix" << endl;
        }
    }

    WFrame(Eigen::MatrixXd d, Eigen::VectorXd v)
    {
        n_samples = d.cols();
        data = d;
        indep_vars = v;
        description = "";
        if (d.cols() > d.rows())
        {
            cov_matrix = cov(data);
        }
        else
        {
            cov_matrix = Eigen::MatrixXd::Ones(1, 1);
            cout << "Warning: Not enough samples to calculate covariance matrix" << endl;
        }
    }

    WFrame(Eigen::MatrixXd d, Eigen::VectorXd v, std::string s)
    {
        n_samples = d.cols();
        data = d;
        indep_vars = v;
        description = s;
        if (d.cols() > d.rows())
        {
            cov_matrix = cov(data);
        }
        else
        {
            cov_matrix = Eigen::MatrixXd::Ones(1, 1);
            cout << "Warning: Not enough samples to calculate covariance matrix" << endl;
        }
    }

    virtual void load(vector<string> fnames) {};
    virtual void trim() {};

    Eigen::VectorXd mean()
    {
        return data.rowwise().mean();
    }

    Eigen::VectorXd cov_diag()
    {

        return cov_matrix.diagonal();
    }

    static Eigen::MatrixXd cov(Eigen::MatrixXd dat)
    {
        const int n_samples = dat.cols();
        Eigen::MatrixXd centered_dat = dat.colwise() - dat.rowwise().mean();
        return centered_dat * centered_dat.transpose() / (n_samples - 1);
    }

    Eigen::MatrixXd get_cov()
    {
        return cov_matrix;
    }

    void set_cov(Eigen::MatrixXd m)
    {
        cov_matrix = m;
    }

    Eigen::MatrixXd get_cov_trunc(Eigen::VectorXd shape)
    {
        int length = shape.sum();
        Eigen::MatrixXd removed_cols = Eigen::MatrixXd(cov_matrix.rows(), length);
        Eigen::MatrixXd trunc_cov = Eigen::MatrixXd(length,length);
        int index = 0;
        for (int i = 0; i < cov_matrix.cols(); i++)
        {
            if (shape(i) == 1)
            {
                removed_cols.col(index) = cov_matrix.col(i);
                index++;
            }
        }

        index = 0;
        for (int i = 0; i < cov_matrix.rows(); i++){
            if(shape(i)==1){
                trunc_cov.row(index) = removed_cols.row(i);
                index++;
            }
        }
        return trunc_cov;
    }

    Eigen::MatrixXd subset(int start, int end){
        if(start < 0 || end >= n_samples || start>end) return Eigen::MatrixXd::Zero(1,1);
        return data(Eigen::placeholders::all,Eigen::seqN(start,end));
    };


    friend ostream &operator<<(std::ostream &os, WFrame const &m)
    {
        os << m.data << endl;
        return os;
    };
};