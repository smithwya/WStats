// Lattice QCD.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<complex>
#include<algorithm>
#include<math.h>
#include<chrono>
#include<random>
#include<fstream>
#include<string>



using namespace std;
typedef std::complex<double> comp;
int rows = 1;
int cols = 1;
int nPoints = 100;


struct statistic {
	double avg;
	double err;

	statistic(double a, double e) {
		avg = a;
		err = e;

	}
	statistic() {
		avg = 0;
		err = 0;
	}
};

void print(statistic st) {
	cout << st.avg << " +/- " << st.err << endl;

}

double average(vector<double> dat) {
	int l = dat.size();
	double sum = 0;

	for (int i = 0; i < l; i++) {
		sum += dat.at(i);
	}

	return sum / l;
}

double jackstddev(vector<double> dat, double avg) {
	int l = dat.size();
	double var = 0;
	for (int i = 0; i < l; i++) {
		var += pow(dat.at(i) - avg, 2);

	}
	var = var * (l - 1) / (l);

	return sqrt(var);
}

statistic jackknife(vector<double> dat) {
	double avg = average(dat);
	int l = dat.size();
	vector<double> thetas = {};

	for (int i = 0; i < l; i++) {
		vector<double> a = dat;
		a.erase(a.begin() + i);
		thetas.push_back(average(a));
		cout<<"ensemble "<<i<<" average is "<<average(a)<<endl;
	}

	double thetatilde = average(thetas);

	double unbiased = avg - (l - 1) * (thetatilde - avg);
	double err = jackstddev(thetas, avg);

	return statistic(avg, err);
}

double stddev(vector<double> dat, double avg) {
	int l = dat.size();
	double var = 0;
	for (int i = 0; i < l; i++) {
		var += pow(dat.at(i) - avg, 2);

	}
	var = var / (l-1);

	return sqrt(var);
}

statistic naiveAnalyze(vector<double> dat) {

	double avg = average(dat);
	double err = stddev(dat, avg);

	return statistic(avg, err);
}

void print(vector<vector<vector<double>>> vec,int layer){
	int rows = vec.size();
	int cols = vec.at(0).size();
	
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			cout<<vec[i][j][layer]<<" ";
		}
		cout<<endl;
	}
	
}

void print(vector<double> v){
	for(int i = 0; i < v.size(); i++){
		
		cout <<v[i]<<" ";
	}
	cout<<endl;
	
}

void print(vector<vector<statistic>> v){
	
		for(int i = 0; i < v.size(); i++){
		for(int j = 0; j < v.at(0).size(); j++){
			cout<<v[i][j].avg<<" ";
		}
		cout<<endl;
	}
	
}


void doit(string folder, string betaxi, string suffix) {
	string collectedname = folder+"/"+betaxi+suffix+".txt";
	string errorname = folder+"/"+betaxi+suffix+"err"+".txt";
	cout<<"collectedname: "<< collectedname<<endl;
	vector<vector<vector<double>>> Data(rows, vector<vector<double>>(cols));
	vector<vector<statistic>> results(rows, vector<statistic>(cols));


	for (int num = 1; num <= nPoints; num++) {
		
		string dataname = folder +"/"+betaxi+"/run" + std::to_string(num)+ suffix;

		std::fstream infile;
		infile.open(dataname, ios::in);

		int size = rows * cols;
		int index = 0;
		double x = 0;

		while (infile >> x) {

			Data.at((index % size) / cols).at((index % size) % cols).push_back(x);
			cout<<x<<endl;
			index++;
		}
		infile.close();
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			results.at(i).at(j) = jackknife(Data.at(i).at(j));
			
		}
	}


	std::ofstream outfile;
	std::ofstream errfile;
	outfile.open(collectedname);
	errfile.open(errorname);
	cout<<collectedname<<endl;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			outfile << results[r][c].avg<< " ";
			errfile << results[r][c].err<< " ";
		}
		outfile << endl;
		errfile << endl;
	}
	outfile.close();
	errfile.close();

	return;
}



int main(int argc, char ** argv)
{
	nPoints = atoi(argv[1]);
	string foldername = argv[2];
	string betaxi = argv[3];
	rows = atoi(argv[4]);
	cols = atoi(argv[5]);
	string suffix = argv[6];
	cout<<"suffix: "<<suffix<<endl;
	doit(foldername, betaxi, suffix);

	return 0;
}
