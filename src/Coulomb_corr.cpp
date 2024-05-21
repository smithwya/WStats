#pragma once
#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"Coulomb_corr.h"
#include"WMath.h"


        Coulomb_corr::Coulomb_corr(int r, int t){
            n_samples = 0;
            description = "empty";
            R_max=r;
            T_max=t;
        }

        

        void Coulomb_corr::load(string fname){
            data = {};
            std::fstream infile;
            int R = 0;
            int T = 0;
            double val=0;

            infile.open(fname);
            cout<<fname<<endl;
            Eigen::MatrixXd temp = -Eigen::MatrixXd::Ones(R_max,T_max);

            while(infile>>R>>T>>val){
                if(R==1 && T==1){
                    data.push_back(temp);
                    temp=-Eigen::MatrixXd::Ones(R_max,T_max);
                }
                temp(R-1,T-1) = val;
            }
            data.push_back(temp);
            infile.close();
            n_samples = data.size();
        }


        ostream& operator<<(ostream& os, Coulomb_corr const& m) {
            for(Eigen::MatrixXd mat : m.data){
                os<<mat<<endl<<endl;
            }
            return os;
        }

        void Coulomb_corr::trim(){
            auto condition = [](const Eigen::MatrixXd mat){
                return (mat.array()==-1).any();
            };
            data.erase(std::remove_if(data.begin(), data.end(), condition), data.end());
            n_samples = data.size();
        }

        void Coulomb_corr::flatten(){
            for(int i = 0; i < n_samples; i++){
                data.at(i).resize(R_max*T_max,1);
            }
        }

        void Coulomb_corr::inflate(){
            for(int i = 0; i < n_samples; i++){
                data.at(i).resize(R_max,T_max);
            }
        }

        Eigen::MatrixXd Coulomb_corr::oneMat(){
            flatten();
            Eigen::MatrixXd flat_mat = Eigen::MatrixXd::Zero(R_max*T_max,n_samples);
            for(int i = 0; i < n_samples; i ++){
                flat_mat.col(i) = data.at(i).array();
            }
            inflate();
            return flat_mat;
        }

        void Coulomb_corr::truncate(int r_min,int r_max, int t_min,int t_max){
            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(r_max-r_min+1,t_max-t_min+1);
            for(int i =0; i < n_samples; i++){
                temp = data.at(i)(Eigen::seq(r_min,r_max),Eigen::seq(t_min,t_max));
                data[i]= temp;
            }
            R_max = r_max-r_min+1;
            T_max = t_max-t_min+1;
            return;
        }

        
