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

namespace WFit
{
    std::vector<double> options = {};
    std::vector<double> params = {};
    std::vector<double> steps = {};
    int num_params = 0;
    ROOT::Math::Minimizer *minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex");
    WModel *model;
    Eigen::MatrixXd *data_frame;

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

    void load_data(Eigen::MatrixXd *d){
        data_frame = d;
    }

    double minfunc(const double *xx)
    {
        double sum = 0;
        Eigen::VectorXd model_result = model->evaluate(xx);
        for(int i = 0; i < data_frame->cols(); i++){
            sum+=(data_frame->col(i)-model_result).squaredNorm();
        }
        return sum;
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
}