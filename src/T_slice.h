#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"
#include"Coulomb_corr.h"

class T_slice : public WFrame{
    public:
    int R;
    int T_max;
    T_slice(Coulomb_corr c, int r,int t_max);
    void jackknife_self();
    friend ostream& operator<<(ostream& os, T_slice const& m);

};