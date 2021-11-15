#include <omp.h>
#include <cmath>
#include "physics.h"
#include "math_aux.h"
#include "param.h"

void CoulombPotential(arma::dmat &V, arma::vec &x,arma::vec &z){
    double soft = 0.65;
    for(int i=0;i<V.n_rows;i++){
        for (int j=0;j<V.n_cols;j++){
            V(i,j) =-1/sqrt(x(i)*x(i)+z(j)*z(j)+soft);
        }
    }
}

void HamX(arma::cx_mat &Hx, arma::dmat &Potential,double VecPot, arma::vec &x, const double dx, const int i){
    arma::cx_colvec d = 1.0/pow(dx,2)*arma::ones<arma::cx_colvec>(x.n_elem) + 0.5*(Potential.col(i))+0.5/(pow(137.04,2))*pow(VecPot,2);
    arma::cx_colvec u = -1.0/(2.0*dx*dx)*arma::ones<arma::cx_colvec>(x.n_elem)+std::complex<double>(0.0,1.0)/(2.0*137.04*dx)*VecPot;
    arma::cx_colvec l = -1.0/(2.0*dx*dx)*arma::ones<arma::cx_colvec>(x.n_elem)-std::complex<double>(0.0,1.0)/(2.0*137.04*dx)*VecPot;

    Hx.col(0) = u;
    Hx.col(1) = d;
    Hx.col(2) = l;

    Hx.col(0)(x.n_elem-1) = 0.0;
    Hx.col(2)(0) = 0.0;
}


void HamX(arma::cx_mat &Hx, arma::dmat &Potential,double VecPot, arma::vec &x, arma::vec &z, const double dx, const double dz, const int i){
    arma::cx_colvec d = 1.0/pow(dz,2)*arma::ones<arma::cx_colvec>(x.n_elem) +0.5*(Potential.col(i))+0.5/(pow(137.04,2))*pow(VecPot,2);
    arma::cx_colvec u = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(x.n_elem)+std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;
    arma::cx_colvec l = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(x.n_elem)-std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;

    Hx.col(0) = u;
    Hx.col(1) = d;
    Hx.col(2) = l;

    Hx.col(0)(x.n_elem-1) = 0.0;
    Hx.col(2)(0) = 0.0;
}


void HamZ(arma::cx_mat &Hz, arma::dmat &Potential,double VecPot, arma::vec &x, arma::vec &z, const double dx, const double dz, const int i){
    arma::cx_colvec d = 1.0/pow(dz,2)*arma::ones<arma::cx_colvec>(z.n_elem) +0.5*(Potential.row(i).t())+0.5/(pow(137.04,2))*pow(VecPot,2);
    arma::cx_colvec u = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)+std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;
    arma::cx_colvec l = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)-std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;

    Hz.col(0) = u;
    Hz.col(1) = d;
    Hz.col(2) = l;

    Hz.col(0)(z.n_elem-1) = 0.0;
    Hz.col(2)(0) = 0.0;
}

void StepX(arma::cx_mat &Mx_dl, arma::cx_mat &Mx_d, arma::cx_mat &Mx_du, arma::cx_mat &Mpx_dl, arma::cx_mat &Mpx_d, arma::cx_mat &Mpx_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut,const int Nx,const int Nz){
    arma::cx_mat PsiNew(Nx,Nz,arma::fill::zeros);
    
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int j=0; j<Nz;j++){
	    arma::cx_mat M(Nx,3,arma::fill::zeros);
        arma::cx_mat Mp(Nx,3,arma::fill::zeros);
	    arma::cx_colvec bCol(Nx);
        arma::cx_colvec PsiColNew(Nx,arma::fill::zeros);
        arma::cx_colvec PsiCol(Nx,arma::fill::zeros);
	    M.col(0) = Mx_dl.col(j);
	    M.col(1) = Mx_d.col(j);
	    M.col(2) = Mx_du.col(j);
	    Mp.col(0) = Mpx_dl.col(j);
	    Mp.col(1) = Mpx_d.col(j);
	    Mp.col(2) = Mpx_du.col(j);
	    PsiCol = Psi.col(j);
	    tridot(Mp,PsiCol,bCol,Nx);
        tdmaSolver(M,bCol,PsiColNew,Nx);
	    PsiNew.col(j) = PsiColNew;
    }
    double end = omp_get_wtime();  
    #ifdef DEBUG
        std::cout<<"[DEBUG] StepZ exectime: "<<(end-start)*1000<<std::endl;
    #endif 
    PsiOut = PsiNew;
}



void StepZ(arma::cx_mat &Mz_dl, arma::cx_mat &Mz_d, arma::cx_mat &Mz_du, arma::cx_mat &Mpz_dl, arma::cx_mat &Mpz_d, arma::cx_mat &Mpz_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut,const int Nx,const int Nz){
    arma::cx_mat PsiNew(Nx,Nz,arma::fill::zeros);
    
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int j=0; j<Nx;j++){
	    arma::cx_mat M(Nz,3,arma::fill::zeros);
        arma::cx_mat Mp(Nz,3,arma::fill::zeros);
	    arma::cx_colvec bCol(Nz);
        arma::cx_colvec PsiColNew(Nz,arma::fill::zeros);
        arma::cx_colvec PsiCol(Nz,arma::fill::zeros);
	    M.col(0) = Mz_dl.col(j);
	    M.col(1) = Mz_d.col(j);
	    M.col(2) = Mz_du.col(j);
	    Mp.col(0) = Mpz_dl.col(j);
	    Mp.col(1) = Mpz_d.col(j);
	    Mp.col(2) = Mpz_du.col(j);
	    PsiCol = Psi.row(j).t();
	    tridot(Mp,PsiCol,bCol,Nz);
        tdmaSolver(M,bCol,PsiColNew,Nz);
	    PsiNew.row(j) = PsiColNew.t();
    }
    double end = omp_get_wtime();  
    #ifdef DEBUG
        std::cout<<"[DEBUG] StepZ exectime: "<<(end-start)*1000<<std::endl;
    #endif 
    PsiOut = PsiNew;
}

std::complex<double> Energy(arma::cx_mat Hx_dl, arma::cx_mat Hx_d,arma::cx_mat Hx_du,arma::cx_mat Hz_dl, arma::cx_mat Hz_d,arma::cx_mat Hz_du,arma::cx_mat &Psi, arma::vec &x, arma::vec &z){
    arma::cx_mat PsiNewX(x.n_elem,z.n_elem,arma::fill::zeros);
    arma::cx_mat PsiNewZ(x.n_elem,z.n_elem,arma::fill::zeros);
    double dx = (x(x.n_elem-1)-x(0))/x.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/z.n_elem;
    
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int j=0; j<z.n_elem;j++){
	    arma::cx_mat Hx(x.n_elem,3,arma::fill::zeros);
        arma::cx_colvec PsiNewXCol(x.n_elem);
        arma::cx_colvec PsiXCol(x.n_elem);

	    Hx.col(0) = Hx_dl.col(j);
	    Hx.col(1) = Hx_d.col(j);
	    Hx.col(2) = Hx_du.col(j);
	    PsiXCol = Psi.col(j);
        tridot(Hx,PsiXCol,PsiNewXCol,x.n_elem);
	    PsiNewX.col(j) = PsiNewXCol;
    }
    #pragma omp parallel for
    for (int j=0; j<x.n_elem;j++){
	    arma::cx_mat Hz(z.n_elem,3,arma::fill::zeros);
        arma::cx_colvec PsiNewZCol(z.n_elem);
        arma::cx_colvec PsiZCol(z.n_elem);

	    Hz.col(0) = Hz_dl.col(j);
	    Hz.col(1) = Hz_d.col(j);
	    Hz.col(2) = Hz_du.col(j);
	    PsiZCol = Psi.row(j).t();
        tridot(Hz,PsiZCol,PsiNewZCol,z.n_elem);
	    PsiNewZ.row(j) = PsiNewZCol.t();
    }

    double end = omp_get_wtime();

    std::complex<double> E = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%(PsiNewZ+PsiNewX)*dx,0)*dz));

    #ifdef DEBUG 
       std::cout<<"[DEBUG] Energy exectime: "<<(end-start)*1000<<"\n";
    #endif
    return E;
}
std::complex<double> AcceX(arma::cx_mat &Psi, arma::dmat &V,double VecPot, arma::vec &x, arma::vec &z){
    int Nx = x.n_elem;
    int Nz = z.n_elem;
    double dx = abs(x(Nx-1)-x(0))/(double)Nx;
    double dz = abs(z(Nz-1)-z(0))/(double)Nz;
    arma::dmat dV(Nx,Nz,arma::fill::zeros);
    derivativeX(V,x,dV);
    std::complex<double> acc = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%(-1*dV)%(Psi)*dx,0)*dz,1));
    return acc;
}

std::complex<double> AcceZ(arma::cx_mat &Psi, arma::dmat &V,double VecPot, arma::vec &x, arma::vec &z){
    int Nx = x.n_elem;
    int Nz = z.n_elem;
    double dx = abs(x(Nx-1)-x(0))/(double)Nx;
    double dz = abs(z(Nz-1)-z(0))/(double)Nz;
    arma::dmat dV(Nx,Nz,arma::fill::zeros);
    derivativeZ(V,z,dV);
    std::complex<double> acc = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%(-1*dV)%(Psi)*dx,0)*dz,1));
    return acc;
}

void maskZ(arma::cx_mat &Mask,arma::vec &x, arma::vec &z, double zb, double gamma){
    int Nx = x.n_elem;
    int Nz = z.n_elem;
    arma::cx_colvec maskvec(Nz,arma::fill::ones);

    for(int i=0;i<Nz;i++){
        if(z(i)<(z(0)+zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(0)+zb))*gamma/(2*zb)),1.0/8.0);

        }

        if(z(i)>(z(Nz-1)-zb)){
            maskvec(i) = pow(cos(M_PI*(z(i)-(z(Nz-1)-zb))*gamma/(2*zb)),1.0/8.0);
        }
    }
    for(int i = 0; i<Nz;i++){
        Mask.col(i) = maskvec(i)*arma::ones<arma::colvec>(Nx);
    }
}
void maskX(arma::cx_mat &Mask,arma::vec &x, arma::vec &z, double xb, double gamma){
    int Nx = x.n_elem;
    int Nz = z.n_elem;
    arma::cx_colvec maskvec(Nx,arma::fill::ones);

    for(int i=0;i<Nx;i++){
        if(x(i)<(x(0)+xb)){
            maskvec(i) = pow(cos(M_PI*(x(i)-(x(0)+xb))*gamma/(2*xb)),1.0/8.0);

        }

        if(x(i)>(x(Nx-1)-xb)){
            maskvec(i) = pow(cos(M_PI*(x(i)-(x(Nx-1)-xb))*gamma/(2*xb)),1.0/8.0);
        }
    }
    for(int i = 0; i<Nx;i++){
        Mask.row(i) = maskvec(i)*arma::ones<arma::rowvec>(Nz);
    }
}
void accelerationMaskX(arma::cx_colvec &accMask, arma::dmat &t, parameters p){
    double period = 2*M_PI/p.w0Ex;
    double start_acc_mask;
    if (p.env==0){
        start_acc_mask = p.fieldPeriods*period;
    }
    else if(p.env==1){
        start_acc_mask = p.fieldPeriods*period + 2.0*period;
    }
    double fwhm = p.fwhm_accMask;
    for (int i = 0; i<accMask.n_elem; i++){
        accMask(i) = 1.0;
        if (t(i)>start_acc_mask){
            accMask(i) = exp(-pow((t(i)-start_acc_mask),2)*fwhm);
        }
    }
}
void accelerationMaskZ(arma::cx_colvec &accMask, arma::dmat &t, parameters p){
    double period = 2*M_PI/p.w0Ez;
    double start_acc_mask;
    if (p.env==0){
        start_acc_mask = p.fieldPeriods*period;
    }
    else if(p.env==1){
        start_acc_mask = p.fieldPeriods*period + 2.0*period;
    }
    double fwhm = p.fwhm_accMask;
    for (int i = 0; i<accMask.n_elem; i++){
        accMask(i) = 1.0;
        if (t(i)>start_acc_mask){
            accMask(i) = exp(-pow((t(i)-start_acc_mask),2)*fwhm);
        }
    }
}
