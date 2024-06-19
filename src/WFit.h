#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>
#include "Math/Minimizer.h"
#include "Minuit2/Minuit2Minimizer.h"
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
    Eigen::VectorXd _sample_avg;

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
        _sample_avg = Eigen::VectorXd::Zero(1);
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
        _sample_avg = _trunc_data.rowwise().mean();
        return;
    };

    //Loads a model
    void set_model(WModel *m)
    {
        _model = m;
        _num_params = m->num_params;
    };
    //loads data frame and sets the inverse covariance matrix to the full one
    void load_data(WFrame *d)
    {
        _data_frame = d;
        _cov_inv = d->cov_matrix.inverse();
    };

    double minfunc_samples(const double *xx)
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

    double minfunc_avg(const double *xx){
        double sum = 0;
        Eigen::VectorXd model_result = _model->evaluate(xx);
        int n_samp = _trunc_data.cols();
        Eigen::VectorXd residual = _sample_avg - model_result;

        return n_samp*residual.transpose()*_cov_inv*residual;

    };

    void minimize()
    {
        truncate_data(_model->data_shape);
        _cov_inv = _data_frame->get_cov_trunc(_model->data_shape).inverse();

        //ROOT::Math::Functor f(this, &WFit::minfunc_avg, _num_params);
        ROOT::Math::Functor f(this, &WFit::minfunc_samples, _num_params);
        minimizer->SetFunction(f);

        for (int i = 0; i < _num_params; i++)
        {
            minimizer->SetVariable(i, to_string(i), _params[i], _steps[i]);
            double var_min = _model->var_lims(i,0);
            double var_max = _model->var_lims(i,1);
            //automatically removes limits if var_min >= var_max
            minimizer->SetVariableLimits(i,var_min,var_max);

        }
        minimizer->Minimize();
    };

    void randomize_start_params(){
        vector<double> pars = {};
        TRandom *r = new TRandom3();
        
        for (int i = 0; i < _num_params; i++)
        {

            double val = 1;
            double var_min = _model->var_lims(i,0);
            double var_max = _model->var_lims(i,1);

            if(var_min<var_max){
                val=var_min+(var_max-var_min)*r->Rndm();
            }
            pars.push_back(val);

        }
        set_params(pars);
        return;
    }


    vector<Eigen::VectorXd> ak_criteria(vector<WModel *> models)
    {

        int n_models = models.size();
        cout<<"fitting "<<n_models<<" models"<<endl;
        Eigen::VectorXd result = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd ak_prob = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd errs = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd statuses = Eigen::VectorXd::Zero(n_models);
        Eigen::VectorXd chisq = Eigen::VectorXd::Zero(n_models);

        int num_dat = _data_frame->data.cols();
        for (int i = 0; i < n_models; i++)
        {

            WModel* ms = models[i];

            set_model(ms);
            int k = ms->num_params;
            int d = ms->data_shape.sum();
            int N_cut = ms->data_shape.size() - d;

            double temp_f = -1.0;


            for(int j = 0; j <10; j++){

                minimizer->Clear();
                set_params(vector<double>(k, 1));
                set_steps(vector<double>(k, 0.1));

                randomize_start_params();

                minimize();
                if(minimizer->MinValue()<temp_f || temp_f<0){
                    ak_prob(i) = minimizer->MinValue() + 2 * k + 2 * N_cut;
                    result(i) = ms->extract_observable(minimizer->X());
                    errs(i) = _model->extract_error(minimizer->X(),minimizer->Errors());
                    statuses(i) = minimizer->Status();

                    temp_f = minimizer->MinValue();
                }
            }

			//if(errs(i)>2.0) ak_prob(i) = ak_prob(i)*1e6;
            //if (statuses(i) > 1 ) ak_prob(i) = ak_prob(i) * 1000000;
            chisq(i) = (minimizer->MinValue()-(num_dat-1)*d)/(d-k);
        }
        int minLoc;
        double min = ak_prob.minCoeff(&minLoc);
        ak_prob = -0.5 * (ak_prob.array() - min);
 
        ak_prob = ak_prob.unaryExpr(&TMath::Exp);
        ak_prob = ak_prob / ak_prob.sum();
        for (int j = 0; j < n_models; j++)
        {
            if (ak_prob(j) < 0.01)
                ak_prob(j) = 0;
        }

        ak_prob = ak_prob / ak_prob.sum();
        Eigen::VectorXd best = Eigen::VectorXd::Zero(1);
        best(0) = chisq(minLoc);
        return {result, errs, ak_prob, chisq, statuses, best};
    };
};
