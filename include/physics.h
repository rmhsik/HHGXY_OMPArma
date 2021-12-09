#ifndef PHYSICS_H
#define PHYSICS_H
#include <complex>
#include <armadillo>

typedef struct parameters parameters;
void CoulombPotential(arma::dmat &V,arma::vec &x, arma::vec &z);
void HamX(arma::cx_mat &Hx, arma::dmat &Potential,double VecPot, double BField, arma::vec &x, arma::vec &z,const double dx, const double dz,const int i);
void HamZ(arma::cx_mat &Hz, arma::dmat &Potential,double VecPot, double BField, arma::vec &x, arma::vec &z,const double dx, const double dz,const int i);
void StepX(arma::cx_mat &Mx_dl, arma::cx_mat &Mx_d, arma::cx_mat &Mx_du, arma::cx_mat &Mpx_dl, arma::cx_mat &Mpx_d, arma::cx_mat &Mpx_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut,const int Nx,const int Nz);
void StepZ(arma::cx_mat &Mz_dl, arma::cx_mat &Mz_d, arma::cx_mat &Mz_du, arma::cx_mat &Mpz_dl, arma::cx_mat &Mpz_d, arma::cx_mat &Mpz_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut,const int Nx,const int Nz);
std::complex<double> Energy(arma::cx_mat Hx_dl, arma::cx_mat Hx_d,arma::cx_mat Hx_du,arma::cx_mat Hz_dl, arma::cx_mat Hz_d,arma::cx_mat Hz_du,arma::cx_mat &Psi, arma::vec &x, arma::vec &z);
std::complex<double> AcceX(arma::cx_mat &Psi, arma::dmat &V,double VecPot, arma::vec &x, arma::vec &z);
std::complex<double> AcceZ(arma::cx_mat &Psi, arma::dmat &V,double VecPot, arma::vec &x, arma::vec &z);
void maskX(arma::cx_mat &Mask,arma::vec &x, arma::vec &z, double zb, double gamma);
void maskZ(arma::cx_mat &Mask,arma::vec &x, arma::vec &z, double zb, double gamma);
void accelerationMaskX(arma::cx_colvec &accMask, arma::dmat &t, parameters p);
void accelerationMaskZ(arma::cx_colvec &accMask, arma::dmat &t, parameters p);
#endif
