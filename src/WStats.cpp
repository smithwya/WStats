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
#include "V_R.h"
#include "Polynomial.h"
#include "nExp_model.h"

using namespace std;
using namespace WMath;
using namespace Eigen;
using namespace WPseudo;

int main(int argc, char **argv)
{
	string fname = argv[1];
	int R_max = atoi(argv[2]);
	int T_max = atoi(argv[3]);
	string save_file = "Fits" + fname.substr(4, fname.size() - 7) + "vr";
	string log_file = "Fits" + fname.substr(4, fname.size() - 7) + "log";
	string sigma_file = "Fits" + fname.substr(4, fname.size() - 7) + "sigma";
	freopen(log_file.c_str(), "w", stdout);

	// Read in the data
	Coulomb_corr G = Coulomb_corr(R_max, T_max);
	G.load(fname);
	// Remove incomplete samples
	G.trim();
	// Bin in R and jackknife
	vector<WFrame> G_R = {};

	for (int i = 0; i < R_max; i++)
	{
		T_slice temp = T_slice(&G, i, T_max);
		G_R.push_back(temp);
	}

	// Generate the set of models to test
	vector<int> n_exps = {1,2,3};
	vector<int> pl_degree_set = {0,1};
	//vector<int> poly_degrees = {2};

	vector<int> t_start_set = {0};
	vector<int> t_end_set = {11};

	VectorXd ind_vars = VectorXd::Zero(T_max);
	for (int i = 0; i < T_max; i++)
	{
		ind_vars(i) = i + 1;
	}

	vector<Exp_model> epoly_models = {};
	vector<nExp_model> nexp_models = {};
	vector<WModel *> mod_ptrs = {};

	for (int pl_d : pl_degree_set)
	{
		for (int t_s : t_start_set)
		{
			for (int t_e : t_end_set)
			{
				VectorXd shape = gen_shape(T_max, t_s, t_e);
				for (int n_e : n_exps)
				{
					nExp_model mod = nExp_model(n_e, pl_d, 2 * n_e + pl_d, ind_vars, shape, "");
					nexp_models.push_back(mod);
				}
				/*
				for (int p_d : poly_degrees)
				{
					Exp_model emod = Exp_model(p_d, pl_d, p_d + pl_d + 1, ind_vars, shape, "");
					epoly_models.push_back(emod);
				}
				*/
			}
		}
	}

	for (int i = 0; i < nexp_models.size(); i++)
	{
		std::cout <<"model "<<i+1<<" "<< nexp_models[i] << endl;
		mod_ptrs.push_back(&nexp_models[i]);
	}

/*
	for (int i = 0; i < epoly_models.size(); i++)
	{
		std::cout << epoly_models[i] << endl;
		mod_ptrs.push_back(&epoly_models[i]);
	}
*/
	int n_models = mod_ptrs.size();


	WFit fitter = WFit();
	fitter.set_options({10000, 1000, .01, 0});
	fitter.set_strat(2);

	VectorXd avg_val = VectorXd::Zero(R_max);
	VectorXd avg_err = VectorXd::Zero(R_max);
	VectorXd best_chi = VectorXd::Zero(R_max);

	for (int R = 0; R < R_max; R++)
	{
		//G_R[R].sparsen(2);
		fitter.load_data(&G_R[R]);
		cout << endl
			 << endl
			 << "Fitting R = " << R + 1 << endl
			 << endl;
		vector<VectorXd> ak_results = fitter.ak_criteria(mod_ptrs);
		std::cout << "Results: " << ak_results[0].transpose() << endl
				  << endl;
		std::cout << "Errors: " << ak_results[1].transpose() << endl
				  << endl;
		std::cout << "Probabilities: " << ak_results[2].transpose() << endl
				  << endl;
		std::cout << "Chisq: " << ak_results[3].transpose() << endl
				  << endl;
		std::cout << "Fit statuses: " << ak_results[4].transpose() << endl
				  << endl;
		std::cout << "Best Chisquared: " << ak_results[5].transpose() << endl
				  << endl;

		for (int i = 0; i < n_models; i++)
		{
			avg_err(R) += pow(ak_results[1](i), 2) * ak_results[2](i);
			avg_err(R) += pow(ak_results[0](i), 2) * ak_results[2](i);
		}

		avg_val(R) = (ak_results[0].array() * ak_results[2].array()).sum();
		avg_err(R) -= pow(avg_val(R), 2);
		avg_err(R) = sqrt(avg_err(R));
		best_chi(R) = ak_results[5](0);
	}

	ofstream pot_file(save_file);
	cout << save_file << endl;
	MatrixXd fresult(R_max, 3);
	fresult.col(0) = avg_val;
	fresult.col(1) = avg_err;
	fresult.col(2) = best_chi;
	pot_file << fresult << endl;
	
	////////////////////////////////////////////R-fits
	
	vector<Polynomial> lin_models_r = {};
	vector<Cornell_model> corn_models_r = {};
	vector<WModel *> mod_ptrs_r = {};

	VectorXd ind_vars_r = VectorXd::Zero(R_max);
	for (int i = 0; i < R_max; i++)
	{
		ind_vars_r(i) = i + 1;
	}

	vector<int> start_r = {0,1,2};
	vector<int> end_r = {8,9,10,11};


	for (int r_s : start_r)
	{
		for (int r_e : end_r)
		{
			VectorXd shape = gen_shape(R_max, r_s, r_e);

			Polynomial p_mod = Polynomial(2, ind_vars_r, shape, "");
			lin_models_r.push_back(p_mod);

			//Cornell_model c_mod = Cornell_model(ind_vars, shape, "");
			//corn_models_r.push_back(c_mod);

		}
	}

	cout<<"models for fitting the R-dependence: "<<endl;
	for (int i = 0; i < lin_models_r.size(); i++)
	{
		mod_ptrs_r.push_back(&lin_models_r[i]);
		cout<<"model "<<i+1<<" "<<lin_models_r[i]<<endl;
	}
	/*
	for (int i = 0; i < corn_models_r.size(); i++)
	{
		mod_ptrs_r.push_back(&corn_models_r[i]);
		cout<<"model "<<i+1<<" "<<corn_models_r[i]<<endl;
	}
	*/
	int n_models_r = mod_ptrs_r.size();

	double avg_sigma = 0;
	double avg_sigma_err = 0;

	fitter.set_options({10000, 1000, .01, 0});
	fitter.set_strat(2);

	WFrame  potential_dat = WFrame(avg_val,ind_vars_r,avg_err);
	fitter.load_data(&potential_dat);

	vector<VectorXd> ak_results_r = fitter.ak_criteria(mod_ptrs_r);

	cout<<endl<<endl<<"Fitting the potential: "<<endl;
	std::cout << "Results: " << ak_results_r[0].transpose() << endl
			  << endl;
	std::cout << "Errors: " << ak_results_r[1].transpose() << endl
			  << endl;
	std::cout << "Probabilities: " << ak_results_r[2].transpose() << endl
			  << endl;
	std::cout << "Chisq: " << ak_results_r[3].transpose() << endl
			  << endl;
	std::cout << "Fit statuses: " << ak_results_r[4].transpose() << endl
			  << endl;

	for (int i = 0; i < n_models_r; i++)
	{
		avg_sigma_err += pow(ak_results_r[1](i), 2) * ak_results_r[2](i);
		avg_sigma_err += pow(ak_results_r[0](i), 2) * ak_results_r[2](i);
	}

	avg_sigma = (ak_results_r[0].array() * ak_results_r[2].array()).sum();
	avg_sigma_err -= pow(avg_sigma, 2);
	avg_sigma_err = sqrt(avg_sigma_err);
	pot_file.close();

	ofstream tension_file(sigma_file);
	tension_file<< avg_sigma<<" "<<avg_sigma_err<<endl;
	tension_file.close();
	

	return 0;
}
