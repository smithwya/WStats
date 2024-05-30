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
	string fname = argv[1];
	double beta = stod(argv[2]);
	int xi = atoi(argv[3]);
	int N = atoi(argv[4]);
	int T = atoi(argv[5]);
	int R_max = atoi(argv[6]);
	int T_max = atoi(argv[7]);
	string save_file = argv[8];
	string log_file = argv[9];

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

	vector<int> exp_degree_set = {1};
	vector<int> pl_degree_set = {0,1};
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
	vector<WModel*> mod_ptrs = {};
	for(int i = 0; i < n_models; i++){
		mod_ptrs.push_back(&models[i]);
	}
	std::ofstream log(log_file);
	
	set_options({100000, 10000, .01, 1});
	VectorXd Vrs = VectorXd::Zero(R_max);
	MatrixXd Ak_prob = MatrixXd::Zero(R_max,n_models);
	MatrixXd Vr_val = MatrixXd::Zero(R_max,n_models);
	MatrixXd Vr_errs = MatrixXd::Zero(R_max,n_models);
	MatrixXd statuses = MatrixXd::Zero(R_max,n_models);
	MatrixXd chisq_p_dof = MatrixXd::Zero(R_max,n_models);

	for(int R = 0; R < R_max; R++){
		log<<"R="<<R<<"parameters: "<<endl;
		load_data(&G_R[R]);
		VectorXd ak = VectorXd::Zero(n_models);
		VectorXd vv = VectorXd::Zero(n_models);
		VectorXd vv_err = VectorXd::Zero(n_models);
		VectorXd stats = VectorXd::Zero(n_models);
		VectorXd chisq = VectorXd::Zero(n_models);
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
			if(stats(i)!=0) ak(i) = ak(i)*1000000;
			chisq(i) = minimizer->MinValue()/((ms->data_shape.sum() - k) * (_data_frame->n_samples - 1));

			for(int j = 0; j<k; j++) log <<minimizer->X()[i]<<" ";
			log<<endl;
        }
        ak = -0.5*(ak.array()-ak.minCoeff());
        ak = ak.unaryExpr(&TMath::Exp);
		for(int j = 0; j < ak.size(); j++) if (ak(j)<0.01) ak(j) = 0;
		Ak_prob.row(R) = ak/ak.sum();
		Vr_val.row(R) = vv;
		Vr_errs.row(R) = vv_err;
		statuses.row(R) = stats;
		chisq_p_dof.row(R) = chisq;

	}


	log<<"Models:"<<endl;
	for(Exp_model e: models){
		log<<e<<endl;
	}
	log<<"Probabilities: "<<endl<<Ak_prob<<endl<<endl;
	log<<"Values: "<<endl<<Vr_val<<endl<<endl;
	log<<"Errors: "<<endl<<Vr_errs<<endl<<endl;
	log<<"Statuses: "<<endl<<statuses<<endl;
	log<<"chisq per dof: "<<endl<<chisq_p_dof<<endl;

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
	log<<"Mode averaged potential and errors: "<<endl<<mdvr<<endl;

	std::ofstream outfile(save_file);
	outfile<<mdvr<<endl;
	outfile.close();
	log.close();
	/*
	vector<Cornell_model> Vr_models = {};

	vector<int> r_start_set = {0,1,2};
	vector<int> r_end_set = {7,8,9,10,11};
	VectorXd ind_vars_r = VectorXd::Zero(R_max);
	for(int i = 0; i < R_max; i++){
		ind_vars_r(i)=i+1;
	}

	for(int r_s : r_start_set){
		for(int r_e : r_end_set){
			VectorXd shape = gen_shape(R_max,r_s,r_e);

			Cornell_model mod = Cornell_model(ind_vars_r,shape,"");
			Vr_models.push_back(mod);
		}
	}
	int n_models_r = models.size();
	vector<WModel*> mod_ptrs_vr = {};
	for(int i = 0; i < n_models_r; i++){
		mod_ptrs.push_back(&Vr_models[i]);
	}
	WFrame vr_frame = WFrame(model_avg_Vr);
	set_options({100000, 10000, .01, 1});
	load_data(&vr_frame);
    set_params(vector<double>(3, 1));
    set_steps(vector<double>(3, 0.5));
	*/
	/*
	int N_vars= 15;
	int N_samples=1000;
	int N_params = 3;

	double params[3] = {2,4,0.3};
	Eigen::VectorXd ind_vars=Eigen::VectorXd(N_vars);
	ind_vars<<1,2,3,4,5,6,7,8,9,10,11,12,13,14,15;
	VectorXd sh = VectorXd::Ones(ind_vars.size());
	sh(1) = 0;
	cout<<sh<<endl;

	Polynomial poly_base = Polynomial(N_params,ind_vars,Eigen::VectorXd::Ones(N_vars),"");

	Polynomial poly = Polynomial(N_params,ind_vars,sh,"");

	Eigen::MatrixXd fakedat = gen_N_gauss_sample(N_samples,poly_base.evaluate(params),VectorXd::Ones(N_vars)*0.1);
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
