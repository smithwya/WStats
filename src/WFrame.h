#pragma once
#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include"WMath.h"

using namespace std;
using namespace WMath;
class WFrame {

    public:
        int n_samples;
        Eigen::MatrixXd data; 
        string description;

        void load(vector<string> fnames);
        void trim();
        friend ostream& operator<<(std::ostream& os, WFrame const& m);

};