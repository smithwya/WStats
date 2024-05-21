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

using namespace std;
using namespace WMath;
using namespace Eigen;
using namespace WFit;

int main(int argc, char **argv)
{
	int R = atoi(argv[1]);
	int T = atoi(argv[2]);
	Coulomb_corr G251 = Coulomb_corr(R, T);
	G251.load({"/home/smithwya/WStats/2.5-1_GRT.txt"});
	G251.truncate(0, 2, 0, 2);
	G251.trim();
	Eigen::MatrixXd fulldat = G251.oneMat();
	cout<<fulldat<<endl;
	return 0;
}
