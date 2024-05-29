#pragma once
#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include"WMath.h"

using namespace std;
using namespace WMath;
class WFrame {

    public:
        int n_samples;
        Eigen::MatrixXd data; 
        string description;
        Eigen::VectorXd indep_vars;

        WFrame(){n_samples = 0;
        data = Eigen::MatrixXd::Zero(1,1);
        indep_vars = Eigen::MatrixXd::Zero(1,1);
        description = "";};

        WFrame(Eigen::MatrixXd d){
            data = d;
            n_samples = d.cols();
            indep_vars = Eigen::VectorXd::Ones(d.cols());
            for(int i = 0; i < d.cols(); i++){
                indep_vars(i) = i+1;
            }
        }

        WFrame(Eigen::MatrixXd d, Eigen::VectorXd v){
            n_samples = d.cols();
            data = d;
            indep_vars = v;
            description = "";
        }

        WFrame(Eigen::MatrixXd d, Eigen::VectorXd v, std::string s){
            n_samples = d.cols();
            data = d;
            indep_vars = v;
            description =s;
        }

        virtual void load(vector<string> fnames){};
        virtual void trim(){};

    Eigen::VectorXd mean(){
        return data.rowwise().mean();
    }

    Eigen::VectorXd cov_diag(){

        return cov().diagonal();
    }

    Eigen::MatrixXd cov(){
        const int n_samples = data.cols();
        Eigen::MatrixXd centered_dat = data.colwise() - data.rowwise().mean();
        return centered_dat * centered_dat.transpose() / (n_samples - 1);
    }

    friend ostream& operator<<(std::ostream& os, WFrame const& m){
        os<<m.data<<endl;
        return os;
    };

};