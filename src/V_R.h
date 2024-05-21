#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"

class V_R : WFrame{
    public:
    int R_max;
    Eigen::VectorXd data;
    Eigen::VectorXd errors;
    V_R(Eigen::VectorXd d, Eigen::VectorXd e, int r);
    void jackknife_self();
    friend ostream& operator<<(ostream& os, V_R const& m);

};