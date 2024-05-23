#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>
#include "Math/Minimizer.h"
#include "WModel.h"
#include "WFrame.h"
#include "WFit.h"
#include <Math/Factory.h>
#include <Math/Functor.h>
#include <Math/WrappedFunction.h>
#include <TRandom1.h>
#include <TRandom2.h>
#include <TRandom3.h>
#include <TRandomGen.h>

namespace WFit
{
    std::vector<double> options = {};
    std::vector<double> params = {};
    std::vector<double> steps = {};
    int num_params = 0;
    ROOT::Math::Minimizer *minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex");
    WModel *model;
    WFrame *data_frame;
    Eigen::MatrixXd cov_inv;

    void set_params(std::vector<double> pars)
    {
        params = pars;
        num_params = params.size();
        return;
    };

    void set_steps(std::vector<double> s){
        steps = s;
        return;
    }

    void set_options(std::vector<double> opts)
    {
        options = opts;
        minimizer->SetMaxFunctionCalls(opts[0]);
        minimizer->SetMaxIterations(opts[1]);
        minimizer->SetTolerance(opts[2]);
        minimizer->SetPrintLevel(opts[3]);
        return;
    };

    void set_model(WModel *m){
        model = m;
    }


    Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd &data)
    {
        const int n_samples = data.cols();
        Eigen::MatrixXd centered_dat = data.colwise() - data.rowwise().mean();
        Eigen::MatrixXd cov = centered_dat * centered_dat.transpose() / (n_samples - 1);
        return cov;
    };

    void load_data(WFrame *d){
        data_frame = d;
        Eigen::MatrixXd cov = sample_covariance(d->data);
    /*
        Eigen::VectorXd diag = cov.diagonal();
        cov = Eigen::MatrixXd::Zero(cov.rows(),cov.cols());
        for(int i = 0; i < cov.rows(); i++){
            cov(i,i) = diag(i);
        }
    */
        cov_inv = cov.inverse();
    }

    double minfunc(const double *xx)
    {
        double sum = 0;
        Eigen::VectorXd model_result = model->evaluate(xx);
        Eigen::VectorXd model_shape = model->shape;
        int n_samp = data_frame->n_samples;

        for(int i = 0; i < data_frame->n_samples; i++){
            Eigen::VectorXd residual = data_frame->data.col(i).array()*model_shape.array()-model_result.array();
            sum+=residual.transpose()*cov_inv*residual;
        }

        return sum/((double)n_samp);
    };




    void minimize()
    {
        ROOT::Math::Functor f(&minfunc, num_params);
        minimizer->SetFunction(f);
        for(int i = 0; i < num_params; i++){
			minimizer->SetVariable(i,to_string(i),params[i],steps[i]);
		}
        minimizer->Minimize();
    };


    Eigen::VectorXd ak_criteria(vector<WModel*> ms){
        int n_models = ms.size();
        Eigen::VectorXd ak = Eigen::VectorXd::Zero(n_models);
        int data_length = ms[0]->shape.size();

        for(int i = 0; i < n_models; i++){
            set_model(ms[i]);
            int k = ms[i]->num_params;
            // initial guess for parameters
            set_params(vector<double>(k,1));
            // initial step sizes
            set_steps(vector<double>(k,0.5));

            minimize();
            int N_cut = data_length-ms[i]->shape.sum();
            ak(i) = minimizer->MinValue()+2*k + 2*N_cut;
        }
        return ak;
    };

    Eigen::VectorXd chisq_per_dof(vector<WModel*> ms){
        int n_models = ms.size();
        Eigen::VectorXd ak = Eigen::VectorXd::Zero(n_models);
        int data_length = ms[0]->shape.size();

        for(int i = 0; i < n_models; i++){
            set_model(ms[i]);
            int k = ms[i]->num_params;
            // initial guess for parameters
            set_params(vector<double>(k,1));
            // initial step sizes
            set_steps(vector<double>(k,0.5));

            minimize();
            int N_cut = data_length-ms[i]->shape.sum();
            ak(i) = minimizer->MinValue()/(data_frame->data.cols()-k);
        }
        return ak;
    };

}