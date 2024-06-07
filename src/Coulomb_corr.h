#pragma once
#include<iostream>
#include<vector>
#include<complex>
#include<Eigen/Dense>
#include<string>
#include<fstream>
#include"WFrame.h"
#include"WMath.h"

class Coulomb_corr : public WFrame{
    private:
    int R_max;
    int T_max;
    public:
        vector<Eigen::MatrixXd> data_list;
        Coulomb_corr(int r, int t):WFrame(){
            R_max=r;
            T_max=t;
        }

        

        void load(string fname){
            data = {};
            std::fstream infile;
            int R = 0;
            int T = 0;
            double val=0;

            infile.open(fname);
            cout<<"Loaded "<<fname<<endl;
            Eigen::MatrixXd temp = -Eigen::MatrixXd::Ones(R_max,T_max);

            while(infile>>R>>T>>val){
                if(R==1 && T==1){
                    data_list.push_back(temp);
                    temp=-Eigen::MatrixXd::Ones(R_max,T_max);
                }
                temp(R-1,T-1) = val;
            }
            data_list.push_back(temp);
            infile.close();
            n_samples = data.size();
        }

        void trim(){

            auto condition = [](const Eigen::MatrixXd mat){
                return (mat.array()==-1).any();
            };
            int length = data_list.size();

            for(int i = 0; i < data_list.size(); i++){
                if((data_list[i].array()==-1).any()){
                    data_list.erase(data_list.begin()+i);
                    i--;
                    length--;
                }
            }
            n_samples = length;
        }

        void flatten(){
            for(int i = 0; i < n_samples; i++){
                data_list.at(i).resize(R_max*T_max,1);
            }
        }

        void inflate(){
            for(int i = 0; i < n_samples; i++){
                data_list.at(i).resize(R_max,T_max);
            }
        }

        Eigen::MatrixXd oneMat(){
            flatten();
            Eigen::MatrixXd flat_mat = Eigen::MatrixXd::Zero(R_max*T_max,n_samples);
            for(int i = 0; i < n_samples; i ++){
                flat_mat.col(i) = data_list.at(i).array();
            }
            inflate();
            return flat_mat;
        }

        void truncate(int r_min,int r_max, int t_min,int t_max){
            Eigen::MatrixXd temp = Eigen::MatrixXd::Zero(r_max-r_min+1,t_max-t_min+1);
            for(int i =0; i < n_samples; i++){
                temp = data_list.at(i)(Eigen::seq(r_min,r_max),Eigen::seq(t_min,t_max));
                data_list[i]= temp;
            }
            R_max = r_max-r_min+1;
            T_max = t_max-t_min+1;
            return;
        }



};