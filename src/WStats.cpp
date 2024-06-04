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
	double beta = stod(argv[2]);
	int xi = atoi(argv[3]);
	int N = atoi(argv[4]);
	int T = atoi(argv[5]);
	int R_max = atoi(argv[6]);
	int T_max = atoi(argv[7]);
	string save_file = argv[8];
	string log_file = argv[9];

	
	/*
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
		//cout<<G_R[i].data.rows()<<" "<<G_R[0].data.cols()<<endl;
		//cout<<G_R[i].data.col(143)<<endl<<endl;
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
					cout << mod << endl;
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

	WFit fitter = WFit();
	fitter.set_options({100000, 10000, .01, 1});

	WFrame temp_frame = WFrame(G_R[0].subset(0,200));

	fitter.load_data(&temp_frame);


	fitter.set_strat(2);
	vector<VectorXd> ak_results = fitter.ak_criteria(mod_ptrs);
	
	for(int i = 0; i < n_models; i++){

		cout<<"Model: "<< models[i]<<endl;
		}
	}

	
	cout<<"Results: "<< ak_results[0].transpose()<<endl<<endl;
	cout<<"Errors: "<<ak_results[1].transpose()<<endl<<endl;
	cout<<"Probabilities: "<<ak_results[2].transpose()<<endl<<endl;
	cout<<"Chisq/dof: "<<ak_results[3].transpose()<<endl<<endl;
	cout<<"Fit statuses: "<<ak_results[4].transpose()<<endl<<endl;
	
	*/
	
	int N_vars= 10;
	int N_samples=100;
	int N_params = 3;
	WFit fitter = WFit();
	double params[3] = {2,4,0.3};
	Eigen::VectorXd ind_vars=Eigen::VectorXd(N_vars);
	ind_vars<<1,2,3,4,5,6,7,8,9,10;
	VectorXd sh = VectorXd::Ones(ind_vars.size());
	sh(2)=0;
	sh(3)=0;
	//cout<<sh<<endl;

	Polynomial poly_base = Polynomial(N_params,ind_vars,Eigen::VectorXd::Ones(N_vars),"");

	Polynomial poly = Polynomial(N_params,ind_vars,sh,"");

	Eigen::MatrixXd fakedat = gen_N_gauss_sample(N_samples,poly_base.evaluate(params),VectorXd::Ones(N_vars)*0.3);
	WFrame test_frame(fakedat);
	fitter.set_strat(2);
	fitter.load_data(&test_frame);
	fitter.set_model(&poly);
	fitter.set_options({100000, 10000, .001, 1});

	cout<<fitter.chisq_per_dof({&poly})<<endl;

	

	return 0;
}
