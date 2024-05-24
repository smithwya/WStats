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


    T_slice(Coulomb_corr *c, int r,int t_max):WFrame(){
        R=r;
        T_max = t_max;
        n_samples = c->data_list[0].size();
        data = Eigen::MatrixXd::Zero(t_max, n_samples);

        for(int i = 0; i < n_samples; i++){
            data.col(i) = c->data_list[i].row(r);
        }

    }

};