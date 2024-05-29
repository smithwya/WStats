#include <iostream>
#include <complex>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include "WMath.h"
#include "WFrame.h"
#include "Coulomb_corr.h"
#include "WFit.h"
#include "T_slice.h"
#include "Exp_model.h"
#include "Cornell_model.h"
#include "WPseudo.h"
#include <TF1.h>
#include"V_R.h"
#include "Polynomial.h"

using namespace std;
using namespace WMath;
using namespace Eigen;
using namespace WFit;
using namespace WPseudo;

int main(int argc, char **argv)
{
	
	/*
	STRATEGY:
	Given N samples of a correlator G(R,T) at fixed beta, xi
	create N jackknife samples G_n(R,T)
	Slice the data in R, so that we are fitting on G_n(T)|fixed R (Coulomb_T_slice)
	Fit the collection of jackknkife samples, use akaike information criteria to select the best model
	Use the best model to individually fit each jackknife sample, extract the part which is linear in T in the exponent and set it = V_n(R)
	Use best model fit parameters to set initial guess, step size
	Fit each jackknife sample to Cornell, weighting by the estimated error in V_n(R) from MINUIT
	Use Jackknife stat stuff to estimate the string tension

	Need to separately calculate the wilson string tension in the same units to divide
	*/
	
	string fname = argv[1];
	double beta = stod(argv[2]);
	int xi = atoi(argv[3]);
	int N = atoi(argv[4]);
	int T = atoi(argv[5]);
	int R_max = atoi(argv[6]);
	int T_max = atoi(argv[7]);

	//Read in the data
	Coulomb_corr G = Coulomb_corr(R_max,T_max);
	G.load(fname);
	//Remove incomplete samples
	G.trim();
	//Bin in R and jackknife
	vector<WFrame> G_R = {};

	for(int i = 0; i < R_max; i++){
		T_slice temp = T_slice(&G,i,T_max);
		G_R.push_back(temp);
	}
	
	//Generate the set of models to test
	vector<Exp_model> models = {};

	vector<int> exp_degree_set = {2};
	vector<int> pl_degree_set = {0,1,2};
	vector<int> t_start_set = {0};
	vector<int> t_end_set = {11};

	VectorXd ind_vars = VectorXd::Zero(T_max);
	for(int i = 0; i < T_max; i++){
		ind_vars(i)=i+1;
	}

	for(int exp_d : exp_degree_set){
		for(int pl_d : pl_degree_set){
			for(int t_s : t_start_set){
				for(int t_e : t_end_set){
					VectorXd shape = gen_shape(T_max,t_s,t_e);

					Exp_model mod = Exp_model(exp_d, pl_d, exp_d+pl_d+1 ,ind_vars,shape, "");
					models.push_back(mod);
				}
			}
		}
	}
	int n_models = models.size();
	MatrixXd Fs = MatrixXd::Zero(R_max,n_models);

	vector<WModel*> mod_ptrs = {};
	for(int i = 0; i < n_models; i++){
		mod_ptrs.push_back(&models[i]);
	}

	set_options({1000, 1000, .01, 0});
	VectorXd Vrs = VectorXd::Zero(R_max);
	MatrixXd Ak_prob = MatrixXd::Zero(R_max,n_models);
	MatrixXd Vr_val = MatrixXd::Zero(R_max,n_models);
	MatrixXd Vr_errs = MatrixXd::Zero(R_max,n_models);
	MatrixXd statuses = MatrixXd::Zero(R_max,n_models);

	for(int R = 0; R < R_max; R++){
		load_data(&G_R[R]);
		VectorXd ak = VectorXd::Zero(n_models);
		VectorXd vv = VectorXd::Zero(n_models);
		VectorXd vv_err = VectorXd::Zero(n_models);
		VectorXd stats = VectorXd::Zero(n_models);
		for (int i = 0; i < n_models; i++)
        {
			WModel* ms = mod_ptrs[i];
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
        }
        ak = -0.5*(ak.array()-ak.minCoeff());
        ak = ak.unaryExpr(&TMath::Exp);
		Ak_prob.row(R) = ak/ak.sum();
		Vr_val.row(R) = vv;
		Vr_errs.row(R) = vv_err;
		statuses.row(R) = stats;

	}
	cout<<"Models:"<<endl;
	for(Exp_model e: models){
		cout<<e<<endl;
	}
	cout<<"Probabilities: "<<endl<<Ak_prob<<endl<<endl;
	cout<<"Values: "<<endl<<Vr_val<<endl<<endl;
	cout<<"Errors: "<<endl<<Vr_errs<<endl<<endl;
	cout<<"Statuses: "<<endl<<statuses<<endl;

	Eigen::VectorXd model_avg_Vr = Eigen::VectorXd::Zero(R_max);
	Eigen::VectorXd model_avg_err=Eigen::VectorXd::Zero(R_max);

	for(int i = 0; i < R_max; i++){
		model_avg_Vr(i) = (Ak_prob.row(i).array()*Vr_val.row(i).array()).sum();
		double err = (Vr_errs.row(i).cwiseAbs2().array()*Ak_prob.row(i).array()).sum();
		err+= (Vr_val.row(i).cwiseAbs2().array()*Ak_prob.row(i).array()).sum();
		err-=pow(model_avg_Vr(i),2);
		model_avg_err(i) = sqrt(err);
	}
	Eigen::MatrixXd mdvr = Eigen::MatrixXd::Zero(R_max,2);
	mdvr.col(0) = model_avg_Vr;
	mdvr.col(1) = model_avg_err;
	cout<<"Mode averaged potential and errors: "<<endl<<mdvr<<endl;
	
	/*
	int N_vars= 10;
	int N_samples=100;
	int N_params = 3;

	double params[3] = {2,3,1.5};
	Eigen::VectorXd ind_vars=Eigen::VectorXd(N_vars);
	ind_vars<<1,2,3,4,5,6,7,8,9,10;
	VectorXd sh = VectorXd::Ones(ind_vars.size());
	cout<<sh<<endl;

	Polynomial poly_base = Polynomial(N_params,ind_vars,Eigen::VectorXd::Ones(N_vars),"");

	Polynomial poly = Polynomial(N_params,ind_vars,sh,"");

	Eigen::MatrixXd fakedat = gen_N_gauss_sample(N_samples,poly_base.evaluate(params),VectorXd::Ones(N_vars)*0.001);
	WFrame test_frame(fakedat);

	load_data(&test_frame);
	set_model(&poly);
	set_options({100000, 10000, .001, 1});

	cout<<chisq_per_dof({&poly})<<endl;
	*/


	/*
	set_params({1,1,1});
	set_steps({0.1,0.1,0.1});
	minimize();
	*/
	return 0;
}
