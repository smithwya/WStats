#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"

class V_R : public WFrame{
    public:
    int R_max;
    V_R() : WFrame(){};
    V_R(Eigen::VectorXd dat){
        data = dat;
        n_samples = 1;
        description="";
    }

};