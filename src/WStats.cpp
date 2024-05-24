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

	
	for(int exp_d : exp_degree_set){
		for(int pl_d : pl_degree_set){
			for(int t_s : t_start_set){
				for(int t_e : t_end_set){
					VectorXd shape = gen_shape(T_max,t_s,t_e);
					Exp_model mod = Exp_model(exp_d,pl_d,exp_d+pl_d+1,shape,"");
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

	set_options({100000, 10000, .001, 1});
	VectorXd Vrs = VectorXd::Zero(R_max);

	for(int R = 0; R < R_max; R++){
		load_data(&G_R[R]);
		Fs.row(R)=chisq_per_dof(mod_ptrs);
		Vrs[R]=(minimizer->X()[1]);

	}
	cout<<"Chisq/ndof for full cov matrix"<<endl;
	cout<<Fs<<endl;
	for(Exp_model e: models){
		cout<<e<<endl;
	}
	
	/*
	int N_vars= 15;
	int N_samples=100;
	int N_params = 3;

	double params[3] = {2,3,1.5};
	VectorXd sh = VectorXd::Ones(N_vars);
	Polynomial poly = Polynomial(N_params,sh,"");


	Eigen::MatrixXd fakedat = gen_N_gauss_sample(N_samples,poly.evaluate(params),VectorXd::Ones(N_vars)*0.001);
	cout<<fakedat<<endl;
	WFrame test_frame(fakedat);

	load_data(&test_frame);
	set_model(&poly);
	set_options({100000, 10000, .001, 1});
	cout<<chisq_per_dof({&poly})<<endl;
	*/
	return 0;
}
