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
    ROOT::Math::Minimizer *minimizer = ROOT::Math::Factory::CreateMinimizer("Minuit2", "");
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

        _cov_inv = _data_frame->get_cov_trunc(_model->data_shape).inverse();

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
        ak = -0.5 * (ak.array() - ak.minCoeff());
        ak = ak.unaryExpr(&TMath::Exp);
        return ak / ak.sum();
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

            double ndof = (data_length) * (num_samples - 1);
            chisq(i) = minimizer->MinValue() / ndof;
        }
        return chisq;
    };

    /*
    void model_average()
    {
        vector<Exp_model> models = {};

        vector<int> exp_degree_set = {1, 2};
        vector<int> pl_degree_set = {0, 1, 2};
        vector<int> t_start_set = {0, 1, 2};
        vector<int> t_end_set = {9, 10, 11};

        VectorXd ind_vars = VectorXd::Zero(T_max);
        for (int i = 0; i < T_max; i++)
        {
            ind_vars(i) = i + 1;
        }

        for (int exp_d : exp_degree_set)
        {
            for (int pl_d : pl_degree_set)
            {
                for (int t_s : t_start_set)
                {
                    for (int t_e : t_end_set)
                    {
                        VectorXd shape = gen_shape(T_max, t_s, t_e);

                        Exp_model mod = Exp_model(exp_d, pl_d, exp_d + pl_d + 1, ind_vars, shape, "");
                        models.push_back(mod);
                    }
                }
            }
        }
        int n_models = models.size();
        vector<WModel *> mod_ptrs = {};
        for (int i = 0; i < n_models; i++)
        {
            mod_ptrs.push_back(&models[i]);
        }
        std::ofstream log(log_file);

        set_options({100000, 10000, .01, 1});
        VectorXd Vrs = VectorXd::Zero(R_max);
        MatrixXd Ak_prob = MatrixXd::Zero(R_max, n_models);
        MatrixXd Vr_val = MatrixXd::Zero(R_max, n_models);
        MatrixXd Vr_errs = MatrixXd::Zero(R_max, n_models);
        MatrixXd statuses = MatrixXd::Zero(R_max, n_models);
        MatrixXd chisq_p_dof = MatrixXd::Zero(R_max, n_models);

        for (int R = 0; R < R_max; R++)
        {
            log << "R=" << R << "parameters: " << endl;
            load_data(&G_R[R]);
            VectorXd ak = VectorXd::Zero(n_models);
            VectorXd vv = VectorXd::Zero(n_models);
            VectorXd vv_err = VectorXd::Zero(n_models);
            VectorXd stats = VectorXd::Zero(n_models);
            VectorXd chisq = VectorXd::Zero(n_models);
            for (int i = 0; i < n_models; i++)
            {
                WModel *ms = mod_ptrs[i];
                set_model(ms);
                int k = ms->num_params;
                // initial guess for parameters
                set_params(vector<double>(k, 1));
                // initial step sizes
                set_steps(vector<double>(k, 0.5));

                minimize();
                int N_cut = ms->data_shape.size() - ms->data_shape.sum();
                ak(i) = minimizer->MinValue() + 2 * k + 2 * N_cut;
                vv(i) = minimizer->X()[1];
                vv_err(i) = minimizer->Errors()[1];
                stats(i) = minimizer->Status();
                if (stats(i) != 0)
                    ak(i) = ak(i) * 1000000;
                chisq(i) = minimizer->MinValue() / ((ms->data_shape.sum() - k) * (_data_frame->n_samples - 1));

                for (int j = 0; j < k; j++)
                    log << minimizer->X()[i] << " ";
                log << endl;
            }
            ak = -0.5 * (ak.array() - ak.minCoeff());
            ak = ak.unaryExpr(&TMath::Exp);
            for (int j = 0; j < ak.size(); j++)
                if (ak(j) < 0.01)
                    ak(j) = 0;
            Ak_prob.row(R) = ak / ak.sum();
            Vr_val.row(R) = vv;
            Vr_errs.row(R) = vv_err;
            statuses.row(R) = stats;
            chisq_p_dof.row(R) = chisq;
        }

        log << "T Models:" << endl;
        for (Exp_model e : models)
        {
            log << e << endl;
        }
        log << "Probabilities: " << endl
            << Ak_prob << endl
            << endl;
        log << "Values: " << endl
            << Vr_val << endl
            << endl;
        log << "Errors: " << endl
            << Vr_errs << endl
            << endl;
        log << "Statuses: " << endl
            << statuses << endl;
        log << "chisq per dof: " << endl
            << chisq_p_dof << endl;

        Eigen::VectorXd model_avg_Vr = Eigen::VectorXd::Zero(R_max);
        Eigen::VectorXd model_avg_err = Eigen::VectorXd::Zero(R_max);

        for (int i = 0; i < R_max; i++)
        {
            model_avg_Vr(i) = (Ak_prob.row(i).array() * Vr_val.row(i).array()).sum();
            double err = (Vr_errs.row(i).cwiseAbs2().array() * Ak_prob.row(i).array()).sum();
            err += (Vr_val.row(i).cwiseAbs2().array() * Ak_prob.row(i).array()).sum();
            err -= pow(model_avg_Vr(i), 2);
            model_avg_err(i) = sqrt(err);
        }
        Eigen::MatrixXd mdvr = Eigen::MatrixXd::Zero(R_max, 2);
        mdvr.col(0) = model_avg_Vr;
        mdvr.col(1) = model_avg_err;
        log << "Mode averaged potential and errors: " << endl
            << mdvr << endl;

        std::ofstream outfile(save_file);
        outfile << mdvr << endl;

        vector<Cornell_model> Vr_models = {};

        vector<int> r_start_set = {0, 1, 2, 3};
        vector<int> r_end_set = {7, 8, 9, 10, 11};
        VectorXd ind_vars_r = VectorXd::Zero(R_max);
        for (int i = 0; i < R_max; i++)
        {
            ind_vars_r(i) = i + 1;
        }

        for (int r_s : r_start_set)
        {
            for (int r_e : r_end_set)
            {
                VectorXd shape = gen_shape(R_max, r_s, r_e);

                Cornell_model mod = Cornell_model(ind_vars_r, shape, "");
                Vr_models.push_back(mod);
            }
        }
        log << "R Models:" << endl;
        for (Cornell_model c : Vr_models)
        {
            log << c << endl;
        }

        int n_models_r = models.size();
        vector<WModel *> mod_ptrs_vr = {};
        for (int i = 0; i < n_models_r; i++)
        {
            mod_ptrs.push_back(&Vr_models[i]);
        }
        WFrame vr_frame = WFrame(model_avg_Vr);

        MatrixXd vr_cov = MatrixXd::Zero(R_max, R_max);
        for (int i = 0; i < R_max; i++)
        {
            vr_cov(i, i) = pow(model_avg_err(i), 2);
        }
        vr_frame.set_cov(vr_cov);
        set_options({100000, 10000, .01, 1});
        load_data(&vr_frame);
        set_params(vector<double>(3, 1));
        set_steps(vector<double>(3, 0.5));

        VectorXd ak_r = VectorXd::Zero(n_models_r);
        VectorXd sigma = VectorXd::Zero(n_models_r);
        VectorXd sigma_err = VectorXd::Zero(n_models_r);
        VectorXd stats_r = VectorXd::Zero(n_models_r);
        VectorXd chisq_r = VectorXd::Zero(n_models_r);
        for (int i = 0; i < n_models_r; i++)
        {
            WModel *ms = mod_ptrs[i];
            set_model(ms);
            int k = 3;
            minimize();
            int N_cut = ms->data_shape.size() - ms->data_shape.sum();

            ak_r(i) = minimizer->MinValue() + 2 * k + 2 * N_cut;
            sigma(i) = minimizer->X()[2];
            sigma_err(i) = minimizer->Errors()[2];
            stats_r(i) = minimizer->Status();
            if (stats_r(i) != 0)
                ak_r(i) = ak_r(i) * 1000000;
            chisq_r(i) = minimizer->MinValue() / (ms->data_shape.sum() - k);

            for (int j = 0; j < k; j++)
                log << minimizer->X()[i] << " ";
            log << endl;
        }

        ak_r = -0.5 * (ak_r.array() - ak_r.minCoeff());
        ak_r = ak_r.unaryExpr(&TMath::Exp);
        for (int j = 0; j < ak_r.size(); j++)
            if (ak_r(j) < 0.01)
                ak_r(j) = 0;
        ak_r = ak_r / ak_r.sum();

        double model_avg_sigma = (ak_r.array() * sigma.array()).sum();
        double model_avg_sigma_err = (sigma_err.cwiseAbs2().array() * ak_r.array()).sum();

        model_avg_sigma_err += (sigma.cwiseAbs2().array() * ak_r.array()).sum();
        model_avg_sigma_err -= pow(model_avg_sigma, 2);
        model_avg_sigma_err = pow(model_avg_sigma_err, 0.5);

        log << "Probabilities: " << endl
            << ak_r.transpose() << endl
            << endl;
        log << "String tensions: " << endl
            << sigma.transpose() << endl
            << endl;
        log << "Errors: " << endl
            << sigma_err.transpose() << endl
            << endl;
        log << "Statuses: " << endl
            << stats_r.transpose() << endl;
        log << "chisq per dof: " << endl
            << chisq_r.transpose() << endl;
        log << "Model string tension and errors: " << endl
            << "sigma = " << model_avg_sigma << " +/- " << model_avg_sigma_err << endl;
        outfile.close();
        log.close();
    };
    */
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
