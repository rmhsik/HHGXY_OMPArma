#ifndef TRIDOT_H
#define TRIDOT_h
#include <complex>
#include <armadillo>
	void tridot(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N);
	void tridotBatched(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N, const int m, const int id);
	void tridotBatchedR(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id);
	void tridotBatchedZ(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id);
	void tridot(arma::cx_mat &H, arma::cx_colvec &Psi,arma::cx_colvec &Psiout, const int N);
#endif
