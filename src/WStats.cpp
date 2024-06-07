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
	string save_file = "Fits"+fname.substr(4,fname.size()-7)+"vr";
	string log_file = "Fits"+fname.substr(4,fname.size()-7)+"log";
	
	freopen(log_file.c_str(),"w",stdout);
	
	// Read in the data
	Coulomb_corr G = Coulomb_corr(R_max, T_max);
	G.load(fname);
	// Remove incomplete samples
	G.trim();
	// Bin in R and jackknife
	vector<WFrame> G_R = {};

	for (int i = 0; i < R_max; i++)
	{
		T_slice temp = T_slice(&G, 0, T_max);
		G_R.push_back(temp);
	}

	// Generate the set of models to test
	vector<nExp_model> models = {};
	vector<int> n_exps = {1,2,3};
	vector<int> pl_degree_set = {0,1};
	vector<int> t_start_set = {0};
	vector<int> t_end_set = {11};

	VectorXd ind_vars = VectorXd::Zero(T_max);
	for (int i = 0; i < T_max; i++)
	{
		ind_vars(i) = i + 1;
	}

	for (int n_e : n_exps)
	{
		for (int pl_d : pl_degree_set)
		{
			for (int t_s : t_start_set)
			{
				for (int t_e : t_end_set)
				{
					VectorXd shape = gen_shape(T_max, t_s, t_e);
					nExp_model mod = nExp_model(n_e, pl_d, 2*n_e + pl_d, ind_vars, shape, "");
					models.push_back(mod);
					std::cout << mod << endl;
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

	//WFrame temp_frame = WFrame(G_R[0].subset(0,500));

	for(int i = 0; i < n_models; i++){

		std::cout<<"Model: "<< models[i]<<endl;
		
	}
	WFit fitter = WFit();
	fitter.set_options({100000, 10000, .001, 1});
	fitter.set_strat(2);

	

	for(int R = 0; R< R_max; R++){
	fitter.load_data(&G_R[R]);
	cout<<"Fitting R = "<<R<<endl<<endl;
	vector<VectorXd> ak_results = fitter.ak_criteria(mod_ptrs);
	std::cout<<"Results: "<< ak_results[0].transpose()<<endl<<endl;
	std::cout<<"Errors: "<<ak_results[1].transpose()<<endl<<endl;
	std::cout<<"Probabilities: "<<ak_results[2].transpose()<<endl<<endl;
	std::cout<<"Chisq/dof: "<<ak_results[3].transpose()<<endl<<endl;
	std::cout<<"Fit statuses: "<<ak_results[4].transpose()<<endl<<endl;
	}


	return 0;
}
