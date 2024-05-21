#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"
#include"Coulomb_corr.h"
#include"T_slice.h"


    T_slice::T_slice(Coulomb_corr c, int r, int t_max){
        R=r;
        T_max = t_max;
        n_samples = c.data.size();
        data = Eigen::MatrixXd::Zero(t_max, c.data.size());

        for(int i = 0; i < n_samples; i++){
            data.col(i) = c.data[i].row(r-1);
        }

    }

    void T_slice::jackknife_self(){
        data = jackknife_sample(data);
    }

    ostream& operator<<(ostream& os, T_slice const& m) {
        os<<m.data<<endl;
        return os;
    }