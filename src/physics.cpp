#include <omp.h>
#include <cmath>
#include "physics.h"
#include "math_aux.h"
#include "param.h"

void CoulombPotential(arma::mat &V, arma::vec &r, arma::vec &z){
    for(int i=0;i<V.n_rows;i++){
        for(int j=0;j<V.n_cols;j++){
            V(i,j) =-1/sqrt(r(i)*r(i)+z(j)*z(j));
        }
    }
}


void HamR(arma::cx_mat &Hr, arma::dmat &Potential, double BField, arma::vec &r, const double dr,arma::dmat &R, const int i){
    arma::cx_colvec d = 1/pow(dr,2)*arma::ones<arma::cx_colvec>(r.n_elem) + 0.5*(Potential.col(i)+1.0/8.0*pow(BField,2)*R.col(i)%R.col(i));
    arma::cx_colvec u = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)+1.0/(2.0*r));
    arma::cx_colvec l = -1.0/(2.0*dr)*(1.0/dr*arma::ones<arma::cx_colvec>(r.n_elem)-1.0/(2.0*r));

    Hr.col(0) = u;
    Hr.col(1) = d;
    Hr.col(2) = l;

    Hr.col(0)(0) = -1.0/(dr*dr);
    Hr.col(1)(0) = 1.0/(dr*dr)+0.5*Potential.col(i)(0);
    Hr.col(2)(0) = 0.0;

    Hr.col(0)(r.n_elem-1) = 0.0;
    Hr.col(2)(0) = 0.0;
}

void HamZ(arma::cx_mat &Hz, arma::dmat &Potential,double VecPot, double BField, arma::vec &z, const double dz,arma::dmat &R, const int i){
    arma::cx_colvec d = 1.0/pow(dz,2)*arma::ones<arma::cx_colvec>(z.n_elem) + 0.5*(Potential.row(i).t()+1.0/8.0*pow(BField,2)*R.row(i).t()%R.row(i).t())+0.5/(pow(137.04,2))*pow(VecPot,2);
    arma::cx_colvec u = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)+std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;
    arma::cx_colvec l = -1.0/(2.0*dz*dz)*arma::ones<arma::cx_colvec>(z.n_elem)-std::complex<double>(0.0,1.0)/(2.0*137.04*dz)*VecPot;

    Hz.col(0) = u;
    Hz.col(1) = d;
    Hz.col(2) = l;

    Hz.col(0)(z.n_elem-1) = 0.0;
    Hz.col(2)(0) = 0.0;
}

void StepR(arma::cx_mat &Mr_dl, arma::cx_mat &Mr_d, arma::cx_mat &Mr_du, arma::cx_mat &Mpr_dl, arma::cx_mat &Mpr_d, arma::cx_mat &Mpr_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int Nr,const int Nz){

    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int j=0; j<Nz;j++){
	arma::cx_mat M(Nr,3,arma::fill::zeros);
        arma::cx_mat Mp(Nr,3,arma::fill::zeros);
	arma::cx_colvec bCol(Nr);
        arma::cx_colvec PsiColNew(Nr,arma::fill::zeros);
        arma::cx_colvec PsiCol(Nr,arma::fill::zeros);

	M.col(0) = Mr_dl.col(j);
	M.col(1) = Mr_d.col(j);
	M.col(2) = Mr_du.col(j);
	Mp.col(0) = Mpr_dl.col(j);
	Mp.col(1) = Mpr_d.col(j);
	Mp.col(2) = Mpr_du.col(j);
	PsiCol = Psi.col(j);
	tridot(Mp,PsiCol,bCol,Nr);
        tdmaSolver(M,bCol,PsiColNew,Nr);
	PsiNew.col(j) = PsiColNew;
    }
    double end = omp_get_wtime();  
    #ifdef DEBUG
    	std::cout<<"[DEBUG] StepR exectime: "<<(end-start)*1000<<std::endl;
    #endif
    PsiOut = PsiNew;

}

void StepZ(arma::cx_mat &Mz_dl, arma::cx_mat &Mz_d, arma::cx_mat &Mz_du, arma::cx_mat &Mpz_dl, arma::cx_mat &Mpz_d, arma::cx_mat &Mpz_du,arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int Nr,const int Nz){
    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int j=0; j<Nr;j++){
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

std::complex<double> Energy(arma::cx_mat Hr_dl, arma::cx_mat Hr_d, arma::cx_mat Hr_du, arma::cx_mat Hz_dl, arma::cx_mat Hz_d,arma::cx_mat Hz_du,arma::cx_mat &Psi, arma::dmat &R,arma::vec &r, arma::vec &z){
    arma::cx_mat PsiNewR(r.n_elem,z.n_elem,arma::fill::zeros);
    arma::cx_mat PsiNewZ(r.n_elem,z.n_elem,arma::fill::zeros);
        double dr = (r(r.n_elem-1)-r(0))/r.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/z.n_elem;
    
    double start = omp_get_wtime();
  
    #pragma omp parallel for
    for (int j=0; j<z.n_elem;j++){
	arma::cx_mat Hr(r.n_elem,3,arma::fill::zeros);
        arma::cx_colvec PsiNewRCol(r.n_elem);
        arma::cx_colvec PsiRCol(r.n_elem);

	Hr.col(0) = Hr_dl.col(j);
	Hr.col(1) = Hr_d.col(j);
	Hr.col(2) = Hr_du.col(j);
	PsiRCol =  Psi.col(j);
        tridot(Hr, PsiRCol,PsiNewRCol,r.n_elem);
	PsiNewR.col(j) = PsiNewRCol;
    }

    #pragma omp parallel for
    for (int j=0; j<r.n_elem;j++){
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

    std::complex<double> E = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%(PsiNewR+PsiNewZ)*dr,0)*dz,1));
    #ifdef DEBUG 
       std::cout<<"[DEBUG] Energy exectime: "<<(end-start)*1000<<"\n";
    #endif
    return E;
}

std::complex<double> AcceZ(arma::cx_mat &Psi, arma::dmat &V,double VecPot,double BField, arma::dmat &R, arma::vec &r, arma::vec &z){
    int Nr = r.n_elem;
    int Nz = z.n_elem;
    double dr = abs(r(Nr-1)-r(0))/(double)Nr;
    double dz = abs(z(Nz-1)-z(0))/(double)Nz;
    arma::dmat dV(Nr,Nz,arma::fill::zeros);
    derivativeZ(V,z,dV);
    std::complex<double> acc = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%(-1*dV)%(Psi)*dr,0)*dz,1));
    return acc;
}

void maskZ(arma::cx_mat &Mask,arma::vec &r, arma::vec &z, double zb, double gamma){
    int Nz = z.n_elem;
    int Nr = r.n_elem;
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
        Mask.col(i) = maskvec(i)*arma::ones<arma::colvec>(Nr);
    }
}

void maskR(arma::cx_mat &Mask, arma::vec &r, arma::vec &z, double rb, double gamma){
    int Nr = r.n_elem;
    int Nz = z.n_elem;
    arma::cx_rowvec maskvec(Nr,arma::fill::ones);

    for(int i=0;i<Nr;i++){        
        if(r(i)>(r(Nr-1)-rb)){
            maskvec(i) = pow(cos(M_PI*(r(i)-(r(Nr-1)-rb))*gamma/(2*rb)),1.0/4.0);
        }
    }

    for(int i = 0; i<Nr;i++){
        Mask.row(i) = maskvec(i)*arma::ones<arma::rowvec>(Nz);
    }
}

void accelerationMask(arma::cx_colvec &accMask, arma::dmat &t, parameters p){
    double period = 2*M_PI/p.w0E;
    double start_acc_mask;
    if (p.env==0){
        std::cout<<"Here\n";
        start_acc_mask = p.fieldPeriods*period;
    }
    else if(p.env==1){
        std::cout<<"Here 2\n";
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
