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

class WFit
{

private:
    std::vector<double> _options;
    std::vector<double> _params;
    std::vector<double> _steps;
    int _num_params;
    ROOT::Math::Minimizer *minimizer;
    WModel *_model;
    WFrame *_data_frame;
    Eigen::MatrixXd _cov_inv;
    Eigen::MatrixXd _trunc_data;

public:
    WFit()
    {
        _options = {};
        _params = {};
        _steps = {};
        _num_params = 0;
        minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Minimize");
        _model = NULL;
        _data_frame = NULL;
        _cov_inv = Eigen::MatrixXd::Zero(1, 1);
        _trunc_data = Eigen::MatrixXd::Zero(1, 1);
    }

    void set_strat(int strat){
        minimizer->SetStrategy(strat);
        return;
    }

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

    // truncates loaded data to the currently loaded model
    void truncate_data(Eigen::VectorXd shape)
    {
        int length = shape.sum();
        _trunc_data = Eigen::MatrixXd(length, _data_frame->data.cols());
        int index = 0;

        for (int i = 0; i < shape.size(); i++)
        {

            if (shape(i) == 1)
            {
                _trunc_data.row(index) = _data_frame->data.row(i);
                index++;
            }
        }
        return;
    }

    //Loads a model
    void set_model(WModel *m)
    {
        _model = m;
    }
    //loads data frame and sets the inverse covariance matrix to the full one
    void load_data(WFrame *d)
    {
        _data_frame = d;
        _cov_inv = d->cov_matrix.inverse();
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
        truncate_data(_model->data_shape);
        _cov_inv = _data_frame->get_cov_trunc(_model->data_shape).inverse();

        ROOT::Math::Functor f(this, &WFit::minfunc, _num_params);
        minimizer->SetFunction(f);

        for (int i = 0; i < _num_params; i++)
        {
            minimizer->SetVariable(i, to_string(i), _params[i], _steps[i]);
        }
        minimizer->Minimize();
    };

    const double * minimize_bootstrap(int num_bootstraps, int size_bootstrap)
    {
        vector<const double*> boot_sampled_pars = {};

        ROOT::Math::Functor f(this, &WFit::minfunc, _num_params);
        minimizer->SetFunction(f);

        // freeze covariance matrix
        _cov_inv = _data_frame->get_cov_trunc(_model->data_shape).inverse();

        for (int i = 0; i < num_bootstraps; i++)
        {

        }

        minimizer->Minimize();
        return minimizer->X();
    }

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

            double ndof = (data_length) * (num_samples - 1);
            chisq(i) = minimizer->MinValue() / ndof;
        }
        return chisq;
    };

    vector<Eigen::VectorXd> ak_criteria(vector<WModel *> models)
    {

        int n_models = models.size();
        cout<<"fitting "<<n_models<<" models"<<endl;
        Eigen::VectorXd result = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd ak_prob = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd errs = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd statuses = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd chisq_p_dof = Eigen::VectorXd::Zero(n_models);
        for (int i = 0; i < n_models; i++)
        {
            cout<<"model "<<i<<endl;
            WModel* ms = models[i];
            set_model(ms);
            int k = ms->num_params;
            cout<<k<<" params"<<endl;
            // initial guess for parameters
            set_params(vector<double>(k, 1));
            // initial step sizes
            set_steps(vector<double>(k, 0.5));
            minimize();
            cout<<"minimized"<<endl;
            int N_cut = ms->data_shape.size() - ms->data_shape.sum();
            ak_prob(i) = minimizer->MinValue() + 2 * k + 2 * N_cut;
            result(i) = ms->extract_observable(minimizer->X());
            errs(i) = _model->extract_error(minimizer->X(),minimizer->Errors());
            statuses(i) = minimizer->Status();
            //if (statuses(i) > 1 ) ak_prob(i) = ak_prob(i) * 1000000;
            chisq_p_dof(i) = (minimizer->MinValue() - ((ms->data_shape.sum()) * (_data_frame->n_samples - 1)))/(ms->data_shape.sum()-N_cut-ms->num_params);
        }
        cout<<"fit all models"<<endl;
        ak_prob = -0.5 * (ak_prob.array() - ak_prob.minCoeff());
        ak_prob = ak_prob.unaryExpr(&TMath::Exp);
        ak_prob = ak_prob / ak_prob.sum();
        for (int j = 0; j < n_models; j++)
        {
            if (ak_prob(j) < 0.01)
                ak_prob(j) = 0;
        }

        ak_prob = ak_prob / ak_prob.sum();

        return {result, errs, ak_prob, chisq_p_dof, statuses};
    };
};