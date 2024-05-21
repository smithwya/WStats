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

using namespace std;
using namespace WMath;
using namespace Eigen;
using namespace WFit;

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
	vector<T_slice> G_R = {};
	for(int i = 0; i < R_max; i++){
		G_R.push_back(T_slice(G,i,T_max));
		G_R[i].jackknife_self();
	}

	int T_max = 12;
	//Generate the set of models to test
	//vector<WModel*> models = {};
	vector<Exp_model> models = {};

	vector<double> py_degree_set = {1,2,3};
	vector<double> pl_degree_set = {0,1};
	vector<double> t_start_set = {0,1,2};
	vector<double> t_end_set = {7,8,9};

	for(double py: py_degree_set){
		for(double pl: pl_degree_set){
			for(double t_s: t_start_set){
				for(double t_e:t_end_set){
					Exp_model exp = Exp_model();
					exp.init({py,pl,t_s,t_e,(double)T_max},py+pl+1,"");
					models.push_back(exp);
				}
			}
		}
	}
	for(Exp_model e : models){
	set_model(&e);
	load_data(&test_dat);
	// initial guess for parameters
	set_params({1, 1, 1, 1});
	// initial step sizes
	set_steps({0.5, 0.5, 0.5, 0.5});
	// Max func calls, max iter, tolerance, printlevel
	set_options({100000, 10000, .001, 1});
	minimize();
	}
	return 0;
}
