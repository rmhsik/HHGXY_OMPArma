#ifndef PHYSICS_H
#define PHYSICS_H
#include <complex>
#include <armadillo>

typedef struct parameters parameters;
void CoulombPotential(arma::mat &V, arma::vec &r, arma::vec &z);
void HamR(arma::cx_mat &Hr, arma::dmat &Potential, double BField, arma::vec &r, const double dr,arma::dmat &R, const int i);
void HamZ(arma::cx_mat &Hz, arma::dmat &Potential,double VecPot, double BField, arma::vec &z, const double dz,arma::dmat &R, const int i);
void StepR(arma::cx_mat &Mr_dl, arma::cx_mat &Mr_d, arma::cx_mat &Mr_du, arma::cx_mat &Mpr_dl, arma::cx_mat &Mpr_d, arma::cx_mat &Mpr_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int Nr,const int Nz);
void StepZ(arma::cx_mat &Mz_dl, arma::cx_mat &Mz_d, arma::cx_mat &Mz_du, arma::cx_mat &Mpz_dl, arma::cx_mat &Mpz_d, arma::cx_mat &Mpz_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int Nr,const int Nz);
std::complex<double> Energy(arma::cx_mat Hr_dl, arma::cx_mat Hr_d, arma::cx_mat Hr_du, arma::cx_mat Hz_dl, arma::cx_mat Hz_d,arma::cx_mat Hz_du,arma::cx_mat &Psi, arma::dmat &R,arma::vec &r, arma::vec &z);
std::complex<double> AcceZ(arma::cx_mat &Psi, arma::dmat &V,double VecPot,double BField, arma::dmat &R, arma::vec &r, arma::vec &z);
void maskZ(arma::cx_mat &Mask,arma::vec &r, arma::vec &z, double zb, double gamma);
void maskR(arma::cx_mat &Mask, arma::vec &r, arma::vec &z, double rb, double gamma);
void accelerationMask(arma::cx_colvec &accMask, arma::dmat &t, parameters p);
#endif
