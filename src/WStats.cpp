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
/*
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
	vector<T_slice> G_R = {};
	for(int i = 0; i < R_max; i++){
		G_R.push_back(T_slice(G,i,T_max));
		G_R[i].jackknife_self();
	}

	//Generate the set of models to test
	//vector<WModel*> models = {};
	vector<Exp_model> models = {};

	vector<int> exp_degree_set = {1,2,3};
	vector<int> pl_degree_set = {0,1};
	vector<int> t_start_set = {0,1,2};
	vector<int> t_end_set = {7,8,9};
	Eigen::seq(1,10);
	Exp_model(1,4,10,Eigen::VectorXd::Zero(1),"");

	
	for(int exp_d : exp_degree_set){
		for(int pl_d : pl_degree_set){
			for(int t_s : t_start_set){
				for(int t_e : t_end_set){
					VectorXd shape = gen_shape(T_max,t_s,t_e);
					models.push_back(Exp_model(exp_d,pl_d,T_max,shape,""));
				}
			}
		}
	}
	int R=0;

	set_model(&models[0]);
	cout<<models[0]<<endl;
	load_data(&(G_R[R]));
	// initial guess for parameters
	set_params({1, 1, 1, 1});
	// initial step sizes
	set_steps({0.5, 0.5, 0.5, 0.5});
	// Max func calls, max iter, tolerance, printlevel
	set_options({100000, 10000, .001, 1});
	minimize();
	*/
	Exp_model e= Exp_model(1,0,2,Eigen::VectorXd::Ones(3),"");


	double a[2] = {0,1};
	VectorXd v = VectorXd(2);
	v<<0,1;
	cout<<e.evaluate(v.data())<<endl;
	return 0;
}
