#ifndef TDMA_H
#define TDMA_H
#include<complex>
#include <armadillo>
    void tdmaSolver(double **a, double **b, double **c, double **d, double **out, const int N);
    void tdmaSolverBatch(double **a ,double **b, double **c, double **d, 
                         double **out, const int N,const int m, const int id);
    void tdmaSolverBatchC(std::complex<double> **a ,std::complex<double> **b, std::complex<double> **c, std::complex <double> **d, std::complex<double> **out, const int N,const int m, const int id);

void tdmaSolverBatchZ(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id);
void tdmaSolverBatchR(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id);
 void tdmaSolver(arma::cx_mat &H, arma::cx_colvec &Psi, arma::cx_colvec &Psiout, const int N);


#endif
