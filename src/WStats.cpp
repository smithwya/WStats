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
	int R = atoi(argv[1]);
	int T = atoi(argv[2]);
	Coulomb_corr G251 = Coulomb_corr(R, T);
	G251.load({"./2.5-1_GRT.txt"});
	G251.truncate(0, 2, 0, 2);
	G251.trim();
	cout<<G251<<endl;
	T_slice G251_R1 = T_slice(G251,1,3);
	cout<<G251_R1<<endl;
	G251_R1.jackknife_self();
	cout<<G251_R1<<endl;
	return 0;
}
