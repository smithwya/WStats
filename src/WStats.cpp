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
#include "WFit.h"
#include "WPseudo.h"
#include <TF1.h>
#include "Polynomial.h"


using namespace std;
using namespace WMath;
using namespace Eigen;
using namespace WPseudo;

int main(int argc, char **argv)
{

	int N_vars= 10;
	int N_samples=100;
	int N_params = 3;
	WFit fitter = WFit();
	double params[3] = {2,4,0.3};
	Eigen::VectorXd ind_vars=Eigen::VectorXd(N_vars);
	ind_vars<<1,2,3,4,5,6,7,8,9,10;
	VectorXd sh = VectorXd::Ones(ind_vars.size());

	Polynomial poly_base = Polynomial(N_params,ind_vars,Eigen::VectorXd::Ones(N_vars),"");

	Polynomial poly = Polynomial(N_params,ind_vars,sh,"");

	Eigen::MatrixXd fakedat = gen_N_gauss_sample(N_samples,poly_base.evaluate(params),VectorXd::Ones(N_vars)*0.3);
	WFrame test_frame(fakedat);

	fitter.set_strat(2);
	fitter.load_data(&test_frame);
	fitter.set_model(&poly);
	fitter.set_options({1000, 100, .1, 1});
	fitter.set_params({2.5,4,2});
	fitter.set_steps({0.5,1,0.5});
	fitter.minimize();
	return 0;
}
