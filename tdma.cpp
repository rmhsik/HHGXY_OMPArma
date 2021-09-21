#include <iostream>
#include <cmath>
#include "tdma.h"

void tdmaSolver(double **a ,double **b, double **c, double **d, 
            double **out, const int N){

    double wc;
   for(int i = 1; i<=N-1;i++){
        wc = (*a)[i]/(*b)[i-1];
        (*b)[i] = (*b)[i] - wc*(*c)[i-1];
        (*d)[i] = (*d)[i] - wc*(*d)[i-1];
    }

    (*out)[N-1] = (*d)[N-1]/(*b)[N-1];
    
    for(int i = N-2;i>=0;i--){
        (*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
    }
}


void tdmaSolverBatch(double **a ,double **b, double **c, double **d, 
            double **out, const int N,const int m, const int id){
	if (id<m){
	   double wc;
	   int first = id*N;
       int last = N*id + N;
	   std::cout<<"first: "<<first<<std::endl;
	   std::cout<<"last: "<<last<<std::endl;
	   for(int i = first+1; i<=last-1;i++){
			wc = (*a)[i]/(*b)[i-1];
			(*b)[i] = (*b)[i] - wc*(*c)[i-1];
			(*d)[i] = (*d)[i] - wc*(*d)[i-1];
		}

		(*out)[last-1] = (*d)[last-1]/(*b)[last-1];
		
		for(int i = last-2;i>=first;i--){
			(*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
		}
	}
}


void tdmaSolverBatchC(std::complex<double> **a ,std::complex<double> **b, std::complex<double> **c, std::complex <double> **d, std::complex<double> **out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       int last = N*id + N;
	   for(int i = first+1; i<=last-1;i++){
			wc = (*a)[i]/(*b)[i-1];
			(*b)[i] = (*b)[i] - wc*(*c)[i-1];
			(*d)[i] = (*d)[i] - wc*(*d)[i-1];
		}

		(*out)[last-1] = (*d)[last-1]/(*b)[last-1];
		
		for(int i = last-2;i>=first;i--){
			(*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
		}
	}
}

void tdmaSolverBatchZ(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       	   int last = N*id + N;
	   for(int i = 1; i<=N-1;i++){
			wc = a(i,id)/b(i-1,id);
			b(i,id) = b(i,id) - wc*c(i-1,id);
			d(id,i) = d(id,i) - wc*d(id,i-1);
		}

		out(id,N-1) = d(id,N-1)/b(N-1,id);
		
		for(int i = N-2;i>=first;i--){
			out(id,i) = (d(id,i)-c(i,id)*out(id,i+1))/b(i,id);
		}
	}
}

void tdmaSolverBatchR(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       	   int last = N*id + N;
	   for(int i = 1; i<=N-1;i++){
			wc = a(i,id)/b(i-1,id);
			b(i,id) = b(i,id) - wc*c(i-1,id);
			d(i,id) = d(i,id) - wc*d(i-1,id);
		}

		out(N-1,id) = d(N-1,id)/b(N-1,id);
		
		for(int i = N-2;i>=first;i--){
			out(i,id) = (d(i,id)-c(i,id)*out(i+1,id))/b(i,id);
		}
	}
}
