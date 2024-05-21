#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"

class Coulomb_corr : WFrame{
    private:
    int R_max;
    int T_max;
    public:
        vector<Eigen::MatrixXd> data;
        Coulomb_corr(int r, int t);
        void load(string fname);
        friend ostream& operator<<(ostream& os, Coulomb_corr const& m);
        void trim();
        void flatten();
        void inflate();
        Eigen::MatrixXd oneMat();
        void truncate(int r_min,int r_max, int t_min,int t_max);
};