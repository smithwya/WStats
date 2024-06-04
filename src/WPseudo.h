#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>
#include "Math/Minimizer.h"
#include "WModel.h"
#include "WFrame.h"
#include "WFit.h"
#include <Math/Factory.h>
#include <Math/Functor.h>
#include <Math/WrappedFunction.h>
#include <TRandom1.h>
#include <TRandom2.h>
#include <TRandom3.h>
#include <TRandomGen.h>

namespace WPseudo
{
    TRandom *r0 = new TRandom3();


    Eigen::VectorXd sample_gaussian(Eigen::VectorXd mean, Eigen::VectorXd variance){
        Eigen::VectorXd sample = Eigen::VectorXd::Zero(mean.size());
        
        for(int i = 0; i < mean.size(); i++){
            sample(i) = r0->Gaus(mean(i),variance(i));
        }
        return sample;
    }

    Eigen::MatrixXd gen_N_gauss_sample(int n_samples, Eigen::VectorXd mean, Eigen::VectorXd variance){
        int n_params = mean.size();
        Eigen::MatrixXd pseudo_dat = Eigen::MatrixXd::Zero(n_params,n_samples);

	    for(int i = 0; i < n_samples; i++){
            pseudo_dat.col(i) = sample_gaussian(mean,variance);
	    }
        return pseudo_dat;
    };

    Eigen::MatrixXd boot_resample(const Eigen::MatrixXd* dat, int samp_size){
        
        Eigen::MatrixXd resamp = Eigen::MatrixXd::Zero(dat->rows(),samp_size);
        int index = 0;
        for(int i = 0; i<samp_size; i++){
            index = (r0->Rndm())*dat->cols();
            resamp.col(i) = dat->col(index);
        }
        return resamp;
    }





}