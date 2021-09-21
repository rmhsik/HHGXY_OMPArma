#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include <omp.h>
#include "tdma.h"
#include "tridot.h"
#include "int_simpson.h"
#define ARMA_NO_DEBUG

#define OMP_NUM_THREADS 4
void CoulombPotential(arma::mat &V, arma::vec &r, arma::vec &z){

    for(int i=0;i<V.n_rows;i++){
        for(int j=0;j<V.n_cols;j++){
            V(i,j) =-1/sqrt(r(i)*r(i)+z(j)*z(j));
        }
    }
}

double envelope_sin2(double tmax, double t){
    if (t<tmax){
        return pow(sin(M_PI*t/tmax),2);
    }
    else {
        return 0.0;
    }
}

double envelope_trap(double tmax, double t){
    double w = 0.057;
    double T = 2*M_PI/w;
    if (t<tmax+2.0*t){
         if (t<T){
             return pow(sin(M_PI*t/(2.0*T)),2);
         }
         else if (T<t && t<tmax+T){
             return 1.0;
         }
         else{
             return pow(sin(M_PI*(t-tmax-T)/(2.0*T)+M_PI/2.0),2);
         }
    }
    else{
        return 0.0;
    }
}

double EField(double t){
    double E0=0.067;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    return E0*envelope_sin2(tmax,t)*sin(w*t);
}

double BField(double t){
    double B0=0.00;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    double phi = 0.12*M_PI;
    return B0*envelope_sin2(tmax,t)*sin(w*t+phi);
}

void Gaussian(arma::cx_mat &Psi, arma::vec &r, arma::vec &z, const double r0, const double z0, const double a ){
    for(int i=0;i<Psi.n_rows;i++){
        for(int j=0;j<Psi.n_cols;j++){
            Psi(i,j) = exp(-pow(r(i)-r0,2)/a-pow(z(j)-z0,2)/a);
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

/*void StepR(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V,double VecPot, double BField,arma::vec &r, const double dr, int Nr, arma::dmat &R,
           arma::vec &z, const double dz, int Nz, const double t, const std::complex<double> dt){

    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    arma::cx_mat M(Nr,3,arma::fill::zeros);
    arma::cx_mat Mp(Nr,3,arma::fill::zeros);
    arma::cx_mat Hr(Nr,3,arma::fill::zeros);
    arma::cx_mat Iden(Nr,3,arma::fill::zeros);
    Iden.col(1) = arma::ones<arma::cx_colvec>(Nr);
    arma::cx_colvec b(Nr);
    arma::cx_colvec PsiCol(Nr);
    arma::cx_colvec PsiColNew(Nr);
    int j;

    //#pragma omp parallel for private(j)
    for (j=0;j<Nz;j++){
        HamR(Hr, V, BField, r, dr, R, j);
        PsiCol = Psi.col(j);
        M = Iden-std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        Mp = Iden+std::complex<double>(0.0,1.0)/2.0*Hr*dt;
        //M.save("M.dat",arma::raw_ascii);
        tridot(Mp,PsiCol,b,Nr);
        tdmaSolver(M,b,PsiColNew,Nr);
        PsiNew.col(j) = PsiColNew;
    }
    PsiOut = PsiNew;
}*/


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
        //for (int j=0; j<Nr; j++){
    //    tridot(Mpz_dl,Mpz_d,Mpz_du,Psi,b,Nz,Nr,j);
    //}
    //std::cout<<"Here\n";
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


/*void StepZ(arma::cx_mat &Psi,arma::cx_mat &PsiOut, arma::dmat &V, double VecPot, double BField,arma::vec &r, const double dr, int Nr, arma::dmat &R,
           arma::vec &z, const double dz, int Nz, const double t, const std::complex<double> dt){

    arma::cx_mat PsiNew(Nr,Nz,arma::fill::zeros);
    arma::cx_mat M(Nz,3,arma::fill::zeros);
    arma::cx_mat Mp(Nz,3,arma::fill::zeros);
    arma::cx_mat Hz(Nz,3,arma::fill::zeros);
    arma::cx_mat Iden(Nz,3,arma::fill::zeros);
    Iden.col(1) = arma::ones<arma::cx_colvec>(Nz);
    arma::cx_colvec b(Nz);
    arma::cx_colvec PsiCol(Nz);
    arma::cx_colvec PsiColNew(Nz);
    int j;
    //#pragma omp parallel for private(j)
   
    for (j=0;j<Nr;j++){
        HamZ(Hz, V,VecPot,BField, z, dz, R, j);
        PsiCol = Psi.row(j).t();
        M = Iden+std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        Mp = Iden-std::complex<double>(0.0,1.0)/2.0*Hz*dt;
        tridot(Mp,PsiCol,b,Nz);
        tdmaSolver(M,b,PsiColNew,Nz);
        PsiNew.row(j) = PsiColNew.t();
    }
     //M.save("M.dat",arma::raw_ascii);
    PsiOut = PsiNew;
}*/

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

void derivativeZ(arma::dmat &U, arma::dmat z, arma::dmat &DU){
    int Nz = z.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/(double)Nz;
    
    for(int i=0;i<U.n_rows;i++){
        DU(i,0) = (U(i,1)-U(i,0))/dz;
    }
    for(int i=0;i<U.n_rows;i++){
        DU(i,U.n_cols-1) = (U(i,U.n_cols-1)-U(i,U.n_cols-2))/dz;
    }
        //std::cout<<"lol"<<std::endl;

    for(int i=0;i<U.n_rows;i++){
        for(int j=1; j<U.n_cols-1;j++){
            DU(i,j) = (U(i,j+1)-U(i,j-1))/(2.0*dz);
        }
    }
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
    std::cout<<"lol \n";
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


int main(){
    omp_set_num_threads(OMP_NUM_THREADS);

    double rmin = 0;
    double rmax = 100;
    int Nr = 1000;
    double dr = (rmax-rmin)/Nr;
    double zmin = -120;
    double zmax = 120;
    int Nz = 2400;
    double dz = (zmax-zmin)/Nz;

    double w = 0.057;
    double t0 = 0.0;
    double tmax = 4.0*2.0*M_PI/w;
    double dt = 0.02;
    int Nt = (tmax-t0)/dt;
    int Nsteps = 10;

    std::cout<<"Parameters:\n";
    std::cout<<"-------------\n";
    std::cout<<"rmin: "<<rmin<<std::endl;
    std::cout<<"rmax: "<<rmax<<std::endl;
    std::cout<<"Nr: "<<Nr<<std::endl;
    std::cout<<"dr: "<<dr<<std::endl;
    std::cout<<"zmin: "<<zmin<<std::endl;
    std::cout<<"zmax: "<<zmax<<std::endl;
    std::cout<<"Nz: "<<Nz<<std::endl;
    std::cout<<"dz: "<<dz<<std::endl;
    std::cout<<"tlim: "<<tmax<<std::endl;
    std::cout<<"Nt: "<<Nt<<std::endl;
    std::cout<<"dt: "<<dt<<std::endl;

    std::complex<double> norm;
    std::complex<double> energy;

    arma::dmat t = arma::linspace(t0,tmax,Nt);
    arma::dmat ElectricField = arma::colvec(Nt,arma::fill::zeros);
    arma::dmat MagneticField = arma::colvec(Nt,arma::fill::zeros);
    arma::dmat VecPotential = arma::colvec(Nt,arma::fill::zeros);
    arma::vec r = arma::linspace(rmin,rmax,Nr);
    arma::vec z = arma::linspace(zmin,zmax,Nz);
    arma::mat V(Nr,Nz,arma::fill::zeros);
    arma::mat dV(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Psi(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiOld(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiR(Nr,Nz,arma::fill::zeros);
    arma::dmat Psi2(Nr,Nz,arma::fill::zeros);
    arma::cx_mat PsiZ(Nr,Nz,arma::fill::zeros);
    arma::cx_mat MaskZ(Nr,Nz,arma::fill::zeros);
    arma::cx_mat MaskR(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mask(Nr,Nz,arma::fill::zeros);
    arma::dmat R(Nr,Nz,arma::fill::zeros);
    arma::cx_colvec acc(Nt,arma::fill::zeros);
    arma::cx_colvec normVec(Nt/Nsteps,arma::fill::zeros);
    arma::cx_colvec enerVec(Nt/Nsteps,arma::fill::zeros);

    arma::cx_mat Hr_dl(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Hr_d(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Hr_du(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mr_dl(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mr_d(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mr_du(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mpr_dl(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mpr_d(Nr,Nz,arma::fill::zeros);
    arma::cx_mat Mpr_du(Nr,Nz,arma::fill::zeros);

    arma::cx_mat Hz_dl(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Hz_d(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Hz_du(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mz_dl(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mz_d(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mz_du(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mpz_dl(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mpz_d(Nz,Nr,arma::fill::zeros);
    arma::cx_mat Mpz_du(Nz,Nr,arma::fill::zeros);
    

    r = r+dr/2.0;
    for(int i = 0; i<Nr;i++){
        R.row(i) = r(i)*arma::ones<arma::rowvec>(Nz);
    }
    //R.save("R.dat",arma::raw_ascii);
    //r.save("r.dat",arma::raw_ascii);
    //z.save("z.dat",arma::raw_ascii);


    for(int i=0; i<Nt;i++){
        MagneticField(i) = BField(t(i));
        ElectricField(i) = EField(t(i));
    }

    for(int i=0; i<Nt;i++){
        VecPotential(i) = -137.04*intSimpson(EField,0,t(i),6000);
    }



    CoulombPotential(V,r,z);
    //Gaussian(Psi,r,z,0.0,0.0,4.0);
    //std::complex<double> Norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    //std::cout<< Norm <<std::endl;
    //Psi = Psi/sqrt(Norm);
    Psi.load("PsiGround_120_120.dat",arma::raw_ascii);
    //std::cout<< 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1))<<std::endl;

    //V.save("Coulomb.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    //Psi2.save("PsiProb.dat",arma::raw_ascii);

    //PsiOld = Psi;

    derivativeZ(V,z,dV);
    //dV.save("dV.dat",arma::raw_ascii);

    std::cout<<"Mask\n";
    maskZ(MaskZ,r,z,12.0,1.0);
    std::cout<<"MaskZ\n";
    maskR(MaskR,r,z,10.0,1.0);
    std::cout<<"MaskR\n";
    //MaskZ.save("MaskZ.dat",arma::raw_ascii);
    //MaskR.save("MaskR.dat",arma::raw_ascii);
    norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    std::cout<<"Norm: "<<norm<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
    Mask = MaskZ%MaskR;
    double start = omp_get_wtime();
    for(int i=0;i<Nt;i++){
	double start_step = omp_get_wtime();
	#pragma omp parallel for
	for (int j=0; j<z.n_elem;j++){
	    arma::cx_mat Hr(Nr,3,arma::fill::zeros);
            HamR(Hr,V, MagneticField(i),r,dr,R,j);
            Hr_dl.col(j) = Hr.col(0);
	    Hr_d.col(j) = Hr.col(1);
	    Hr_du.col(j) = Hr.col(2);
	    Mr_dl.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(0)*dt/2.0;
	    Mr_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hr.col(1)*dt/2.0;
	    Mr_du.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(2)*dt/2.0;
	    Mpr_dl.col(j) = std::complex<double>(0.0,1.0)*Hr.col(0)*dt/2.0;
	    Mpr_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hr.col(1)*dt/2.0;
	    Mpr_du.col(j) = std::complex<double>(0.0,1.0)*Hr.col(2)*dt/2.0;

        }
	#pragma omp parallel for
	for (int j=0; j<r.n_elem;j++){
	    arma::cx_mat Hz(Nz,3,arma::fill::zeros);
            HamZ(Hz,V,VecPotential(i),MagneticField(i),z,dz,R,j);
            Hz_dl.col(j) = Hz.col(0);
            Hz_d.col(j) = Hz.col(1);
            Hz_du.col(j) = Hz.col(2);
	    Mz_dl.col(j) = std::complex<double>(0.0,1.0)*Hz.col(0)*dt/4.0;
	    Mz_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hz.col(1)*dt/4.0;
	    Mz_du.col(j) = std::complex<double>(0.0,1.0)*Hz.col(2)*dt/4.0;
	    Mpz_dl.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(0)*dt/4.0;
	    Mpz_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hz.col(1)*dt/4.0;
	    Mpz_du.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(2)*dt/4.0;
        }

        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,Nr,Nz);
        //PsiZ = Psi%MaskZ%MaskR;
        StepR(Mr_dl,Mr_d,Mr_du,Mpr_dl,Mpr_d,Mpr_du,PsiZ,PsiR,Nr,Nz);
        //StepR(PsiZ,PsiR,V,VecPotential(i),MagneticField(i),r,dr,Nr, R,z,dz,Nz,t(i),dt);
        //PsiR = PsiR%MaskZ%MaskR;
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiR,Psi,Nr,Nz);
        Psi = Psi%Mask;

        acc(i) = AcceZ(Psi,V,VecPotential(i),MagneticField(i),R,r,z);
	//normVec(i) = Norm;
	//enerVec(i) = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);
        //PsiOld = PsiR2/sqrt(Norm);
        //std::cout<<"Step: "<<i<<" Norm: "<<Norm<<" Mag: "<<MagneticField(i)<<""<<" Acc: "<<acc(i)<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
        //std::cout<<i<<" of "<< Nt <<std::endl;
	double end_step = omp_get_wtime();
	if (i%Nsteps==0){
	    std::cout<<"[DEBUG] Time from init: "<<(end_step-start)<<"\n";
            norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
            energy = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);
            normVec(i%Nsteps) = norm;
            enerVec(i%Nsteps) = energy;
  	    std::cout<<"Step: "<<i<<" Norm: "<<norm<<" Energy "<< energy<<"\n\n";
        }
    }
    double end = omp_get_wtime();

   std::cout <<"Simulation exectime: "<<(end-start)*1000<<std::endl;
   std::cout <<"Timestep exectime: "<<(end-start)*1000/Nt<<std::endl;
    //std::cout<<"End:\n\tNorm: "<<Norm<<" Energy: "<<Energy(Psi,V,VecPotential(Nt-1),MagneticField(Nt-1),R,r,z)<<std::endl;
    Psi2 = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("PsiEnd1.dat",arma::raw_ascii);
    acc.save("acc1.dat",arma::raw_ascii);
    //normVec.save("normVec1.dat",arma::raw_ascii);
    //enerVec.save("enerVer1.dat",arma::raw_ascii);
    MagneticField.save("MagneticField1.dat",arma::raw_ascii);
    VecPotential.save("VecPotential1.dat",arma::raw_ascii);
    ElectricField.save("ElectricField1.dat",arma::raw_ascii);
    //PsiOld.save("PsiGround.dat",arma::raw_ascii);
    //Hr.save("Hr.dat",arma::raw_ascii);
    //Hz.save("Hz.dat",arma::raw_ascii);

    return 0;
}
