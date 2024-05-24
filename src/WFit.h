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
    std::vector<double> _options = {};
    std::vector<double> _params = {};
    std::vector<double> _steps = {};
    int _num_params = 0;
    ROOT::Math::Minimizer *minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex");
    WModel *_model;
    WFrame *_data_frame;
    Eigen::MatrixXd _cov_inv;

    void set_params(std::vector<double> pars)
    {
        _params = pars;
        _num_params = _params.size();
        return;
    };

    void set_steps(std::vector<double> s){
        _steps = s;
        return;
    }

    void set_options(std::vector<double> opts)
    {
        _options = opts;
        minimizer->SetMaxFunctionCalls(opts[0]);
        minimizer->SetMaxIterations(opts[1]);
        minimizer->SetTolerance(opts[2]);
        minimizer->SetPrintLevel(opts[3]);
        return;
    };

    void set_model(WModel *m){
        _model = m;
    }


    Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd &data)
    {
        const int n_samples = data.cols();
        Eigen::MatrixXd centered_dat = data.colwise() - data.rowwise().mean();
        Eigen::MatrixXd cov = centered_dat * centered_dat.transpose() / (n_samples - 1);

        return cov;
    };

    void load_data(WFrame *d){
        _data_frame = d;
        
        /*
        Eigen::MatrixXd cv = d->cov();
        Eigen::VectorXd diag = cv.diagonal();
        Eigen::MatrixXd diag_cov_inv = cv*0;
        for(int i = 0; i < diag.rows(); i++){
            diag_cov_inv(i,i) = 1/diag(i);
        }
        _cov_inv = diag_cov_inv;
        */
        _cov_inv = d->cov().inverse();
    }

    double minfunc(const double *xx)
    {
        double sum = 0;
        Eigen::VectorXd model_result = _model->evaluate(xx);
        Eigen::VectorXd model_shape = _model->shape;
        int n_samp = _data_frame->n_samples;

        for(int i = 0; i < _data_frame->n_samples; i++){
            Eigen::VectorXd residual = _data_frame->data.col(i).array()*model_shape.array()-model_result.array();
            sum+=residual.transpose()*_cov_inv*residual;
        }

        return sum/((double)n_samp);
    };




    void minimize()
    {
        ROOT::Math::Functor f(&minfunc, _num_params);
        minimizer->SetFunction(f);
        for(int i = 0; i < _num_params; i++){
			minimizer->SetVariable(i,to_string(i),_params[i],_steps[i]);
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
        Eigen::VectorXd chisq = Eigen::VectorXd::Zero(n_models);
        int data_length = ms[0]->shape.sum();

        for(int i = 0; i < n_models; i++){
            set_model(ms[i]);
            int k = ms[i]->num_params;
            // initial guess for parameters
            set_params(vector<double>(k,1));
            // initial step sizes
            set_steps(vector<double>(k,0.5));
            minimize();
            
            double ndof = (data_length-k);

            chisq(i) = minimizer->MinValue()/ndof;
        }
        return chisq;
    };

}