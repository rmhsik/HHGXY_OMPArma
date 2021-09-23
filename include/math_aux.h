#ifndef MATH_AUX_H
#define MATH_AUX_H
#include <complex>
#include <armadillo>
typedef struct parameters parameters;

void Gaussian(arma::cx_mat &Psi, arma::vec &r, arma::vec &z, const double r0, const double z0, const double a );
void Exponential(arma::cx_mat &Psi, arma::vec &r, arma::vec &z, const double r0, const double z0, const double a );
void derivativeZ(arma::dmat &U, arma::dmat z, arma::dmat &DU);
void tdmaSolver(double **a ,double **b, double **c, double **d, double **out, const int N);
void tdmaSolver(arma::cx_mat &H, arma::cx_colvec &Psi, arma::cx_colvec &Psiout, const int N);
void tdmaSolverBatch(double **a ,double **b, double **c, double **d, double **out, const int N,const int m, const int id);
void tdmaSolverBatchC(std::complex<double> **a ,std::complex<double> **b, std::complex<double> **c, std::complex <double> **d, std::complex<double> **out, const int N,const int m, const int id);
void tdmaSolverBatchZ(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id);
void tdmaSolverBatchR(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id);
void tridot(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N);
void tridotBatched(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N,const int m, const int id);
void tridotBatchedZ(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id);
void tridotBatchedR(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id);
void tridot(arma::cx_mat &H, arma::cx_colvec &Psi, arma::cx_colvec &Psiout, const int N);
double intSimpson(double (*func)(double,parameters), double from, double to, int n, parameters p);
#endif
