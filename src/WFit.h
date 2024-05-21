#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include<vector>
#include"WModel.h"
#include"WFrame.h"
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"

namespace WFit{

    
    void set_params(std::vector<double> pars);
    void set_options(std::vector<double> opts);
    double minfunc(const double *xx);
    void minimize();

};