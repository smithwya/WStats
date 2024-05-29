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
    Eigen::MatrixXd _trunc_data;

    void set_params(std::vector<double> pars)
    {
        _params = pars;
        _num_params = _params.size();
        return;
    };

    void set_steps(std::vector<double> s)
    {
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

    void truncate_data(Eigen::VectorXd shape)
    {

        int length = shape.sum();
        _trunc_data = Eigen::MatrixXd(length, _data_frame->n_samples);
        int index = 0;

        for (int i = 0; i < shape.size(); i++)
        {

            if (shape(i) == 1)
            {
                _trunc_data.row(index) = _data_frame->data.row(i);
                index++;
            }
        }
    }

    void set_model(WModel *m)
    {
        _model = m;
        truncate_data(m->data_shape);
    }

    void load_data(WFrame *d)
    {
        _data_frame = d;
    }

    Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd &data)
    {
        const int n_samples = data.cols();
        Eigen::MatrixXd centered_dat = data.colwise() - data.rowwise().mean();
        Eigen::MatrixXd cov = centered_dat * centered_dat.transpose() / (n_samples - 1);

        return cov;
    };

    double minfunc(const double *xx)
    {
        double sum = 0;
        Eigen::VectorXd model_result = _model->evaluate(xx);
        int n_samp = _data_frame->n_samples;

        for (int i = 0; i < n_samp; i++)
        {
            Eigen::VectorXd residual = _trunc_data.col(i).array() - model_result.array();
            sum += residual.transpose() * _cov_inv * residual;
        }

        return sum;
    };

    void minimize()
    {
        ROOT::Math::Functor f(&minfunc, _num_params);
        minimizer->SetFunction(f);
        _cov_inv = sample_covariance(_trunc_data).inverse();

        for (int i = 0; i < _num_params; i++)
        {
            minimizer->SetVariable(i, to_string(i), _params[i], _steps[i]);
        }
        minimizer->Minimize();
    };


    Eigen::VectorXd ak_criteria(vector<WModel *> ms)
    {
        int n_models = ms.size();
        Eigen::VectorXd ak = Eigen::VectorXd::Zero(n_models);

        for (int i = 0; i < n_models; i++)
        {
            set_model(ms[i]);
            int k = ms[i]->num_params;
            // initial guess for parameters
            set_params(vector<double>(k, 1));
            // initial step sizes
            set_steps(vector<double>(k, 0.5));

            minimize();
            int N_cut = ms[i]->data_shape.size() - ms[i]->data_shape.sum();
            ak(i) = minimizer->MinValue() + 2 * k + 2 * N_cut;
        }
        ak = -0.5*(ak.array()-ak.minCoeff());
        ak = ak.unaryExpr(&TMath::Exp);
        return ak/ak.sum();
    };

    Eigen::VectorXd chisq_per_dof(vector<WModel *> ms)
    {
        int n_models = ms.size();
        Eigen::VectorXd chisq = Eigen::VectorXd::Zero(n_models);
        int num_samples = _data_frame->n_samples;
        for (int i = 0; i < n_models; i++)
        {
            int data_length = ms[i]->data_shape.sum();
            set_model(ms[i]);
            int k = ms[i]->num_params;
            // initial guess for parameters
            set_params(vector<double>(k, 1));
            // initial step sizes
            set_steps(vector<double>(k, 0.5));
            minimize();

            double ndof = (data_length - k) * (num_samples - 1);
            chisq(i) = minimizer->MinValue() / ndof;
        }
        return chisq;
    };

    /*
    Eigen::MatrixXd sample_covariance(const Eigen::MatrixXd &data)
    {
        const int n_samples = data.cols();
        Eigen::MatrixXd centered_dat = data.colwise() - data.rowwise().mean();
        Eigen::MatrixXd cov = centered_dat * centered_dat.transpose() / (n_samples - 1);

        return cov;
    };
    */

    /*
    Eigen::MatrixXd cv = d->cov();
    Eigen::VectorXd diag = cv.diagonal();
    Eigen::MatrixXd diag_cov_inv = cv*0;
    for(int i = 0; i < diag.rows(); i++){
        diag_cov_inv(i,i) = 1/diag(i);
    }
    _cov_inv = diag_cov_inv;
    */
}