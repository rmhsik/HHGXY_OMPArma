#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <omp.h>
#include "physics.h"
#include "fields.h"
#include "math_aux.h"
#include "param.h"
#define ARMA_NO_DEBUG

int main(){
    parameters p;
    omp_set_num_threads(OMP_NUM_THREADS);    

    #ifdef TEXTOUTPUT
        std::ofstream outfile;
        outfile.open("output_Evolution.txt");
    #endif
    

    #ifdef TEXTOUTPUT
        outfile<<"Parameters:\n";
        outfile<<"------------\n";
        outfile<<"xmin: "<<p.xmin<<std::endl;
        outfile<<"xmax: "<<p.xmax<<std::endl;
        outfile<<"Nx: "<<p.Nx<<std::endl;
        outfile<<"dx: "<<p.dx<<std::endl;
        outfile<<"zmin: "<<p.zmin<<std::endl;
        outfile<<"zmax: "<<p.zmax<<std::endl;
        outfile<<"Nz: "<<p.Nz<<std::endl;
        outfile<<"dz: "<<p.dz<<std::endl;
        outfile<<"tlim: "<<p.tmax<<std::endl;
        outfile<<"Nt: "<<p.Nt<<std::endl;
        outfile<<"dt: "<<p.dt<<std::endl;

    #else    
        std::cout<<"Parameters:\n";
        std::cout<<"------------\n";
        std::cout<<"xmin: "<<p.xmin<<std::endl;
        std::cout<<"xmax: "<<p.xmax<<std::endl;
        std::cout<<"Nx: "<<p.Nx<<std::endl;
        std::cout<<"dx: "<<p.dx<<std::endl;
        std::cout<<"zmin: "<<p.zmin<<std::endl;
        std::cout<<"zmax: "<<p.zmax<<std::endl;
        std::cout<<"Nz: "<<p.Nz<<std::endl;
        std::cout<<"dz: "<<p.dz<<std::endl;
        std::cout<<"tlim: "<<p.tmax<<std::endl;
        std::cout<<"Nt: "<<p.Nt<<std::endl;
        std::cout<<"dt: "<<p.dt<<std::endl;
    #endif
    std::complex<double> norm;
    std::complex<double> energy;

    arma::vec t = arma::linspace(p.t0,p.tmax, p.Nt);
    arma::vec x = arma::linspace(p.xmin,p.xmax,p.Nx);
    arma::vec z = arma::linspace(p.zmin,p.zmax,p.Nz);
    arma::vec ElectricFieldX(p.Nt,arma::fill::zeros);
    arma::vec ElectricFieldZ(p.Nt,arma::fill::zeros);
    arma::vec MagneticFieldX(p.Nt,arma::fill::zeros);
    arma::vec MagneticFieldZ(p.Nt,arma::fill::zeros);
    arma::vec VecPotentialX(p.Nt,arma::fill::zeros);
    arma::vec VecPotentialZ(p.Nt,arma::fill::zeros);
    arma::dmat V(p.Nx,p.Nz,arma::fill::zeros);
    arma::dmat dV(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Psi(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat PsiOld(p.Nx,p.Nz, arma::fill::zeros);
    arma::dmat Psi2(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat PsiX(p.Nx, p.Nz, arma::fill::zeros);
    arma::cx_mat PsiZ(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat MaskX(p.Nx, p.Nz, arma::fill::zeros);
    arma::cx_mat MaskZ(p.Nx, p.Nz, arma::fill::zeros);
    arma::cx_mat Mask(p.Nx, p.Nz, arma::fill::zeros);
    arma::cx_colvec accX(p.Nt,arma::fill::zeros);
    arma::cx_colvec accZ(p.Nt,arma::fill::zeros);
    arma::cx_colvec accMaskX(p.Nt,arma::fill::zeros);
    arma::cx_colvec accMaskZ(p.Nt,arma::fill::zeros);
    arma::cx_colvec normVec(p.Nt/p.Nsteps,arma::fill::zeros);
    arma::cx_colvec enerVec(p.Nt/p.Nsteps,arma::fill::zeros);
    
    arma::cx_mat Hx_dl(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Hx_d(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Hx_du(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mx_dl(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mx_d(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mx_du(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mpx_dl(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mpx_d(p.Nx,p.Nz, arma::fill::zeros);
    arma::cx_mat Mpx_du(p.Nx,p.Nz, arma::fill::zeros);

    arma::cx_mat Hz_dl(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Hz_d(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Hz_du(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mz_dl(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mz_d(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mz_du(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mpz_dl(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mpz_d(p.Nz,p.Nx, arma::fill::zeros);
    arma::cx_mat Mpz_du(p.Nz,p.Nx, arma::fill::zeros);
    
    #ifdef TEXTOUTPUT
        outfile<<"Matrices definition\n";
    #else
        std::cout<<"Matrices definition\n";
    #endif

    CoulombPotential(V,x,z);
    Psi.load("results/PsiGround.dat",arma::raw_ascii);

    #ifdef TEXTOUTPUT
        outfile<<"Coulomb, Gaussian\n";
    #else
        std::cout<<"Coulomb, Gaussian\n";
    #endif

    for(int i=0; i<p.Nt;i++){
        ElectricFieldX(i) = EFieldX(t(i),p);
        ElectricFieldZ(i) = EFieldZ(t(i),p);
        MagneticFieldX(i) = BFieldX(t(i),p);
        MagneticFieldZ(i) = BFieldZ(t(i),p);
    }

    #ifdef TEXTOUTPUT
        outfile<<"Fields\n";
    #else
        std::cout<<"Fields\n";
    #endif

    for(int i=0; i<p.Nt;i++){
        VecPotentialX(i) = -137.04*intSimpson(EFieldX,0,t(i),6000,p);
        VecPotentialZ(i) = -137.04*intSimpson(EFieldZ,0,t(i),6000,p);
    }

    #ifdef TEXTOUPUT
        outfile<<"VecPotential \n";
    #else
        std::cout<<"VecPotential \n";
    #endif

    norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
    std::cout<<norm<<std::endl;
    Psi = Psi/sqrt(norm);
    norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
    
    #ifdef TEXTOUTPUT
        outfile<<norm<<std::endl;
        outfile<<"Norm\n";
    #else
        std::cout<<norm<<std::endl;
        std::cout<<"Norm\n";
    #endif

    maskZ(MaskZ,x,z,12.0,1.0);
    maskX(MaskX,x,z,12.0,1.0);
    Mask = MaskZ%MaskX;
    accelerationMaskX(accMaskX,t,p);
    accelerationMaskZ(accMaskZ,t,p); 
    #ifdef TEXTOUTPUT
        outfile<<"Mask\n";
    #else
        std::cout<<"Mask\n";
    #endif

    arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
    arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
    for(int i=0;i<p.Nx;i++){
        HamZ(Hz,V,0.0,0.0,x,z,p.dx,p.dz,i);
        Hz_dl.col(i) = Hz.col(0);
        Hz_d.col(i) = Hz.col(1);
        Hz_du.col(i) = Hz.col(2);
    }
    #ifdef TEXTOUPUT
        outfile<<"HamZ\n";
    #else
        std::cout<<"HamZ\n";
    #endif

    for(int i=0;i<p.Nz;i++){
        HamX(Hx,V,0.0,0.0,x,z,p.dx,p.dz,i);
        Hx_dl.col(i) = Hx.col(0);
        Hx_d.col(i) = Hx.col(1);
        Hx_du.col(i) = Hx.col(2);
    }
    #ifdef TEXTOUTPUT
        outfile<<"HamX\n";
    #else
        std::cout<<"HamX\n";
    #endif
   
    energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
    #ifdef TEXOUTPUT
        outfile<<"Energy: "<<energy<<std::endl;
    #else
        std::cout<<"Energy: "<<energy<<std::endl;
    #endif
    double dt = p.dt;

    for (int i=0; i<p.Nt;i++){
        for(int j=0;j<p.Nz;j++){
            arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
            HamX(Hx,V,VecPotentialX(i),MagneticFieldZ(i),x,z,p.dx,p.dz,j);
            Hx_dl.col(j) = Hx.col(0);
            Hx_d.col(j) = Hx.col(1);
            Hx_du.col(j) = Hx.col(2);

            Mx_dl.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mx_d.col(j)  = 1-std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mx_du.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;

            Mpx_dl.col(j) = std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mpx_d.col(j)  = 1+std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mpx_du.col(j) = std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;
        }
        for(int j=0;j<p.Nx;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,VecPotentialZ(i),MagneticFieldZ(i),x,z,p.dx,p.dz,j);
            Hz_dl.col(j) = Hz.col(0);
            Hz_d.col(j) = Hz.col(1);
            Hz_du.col(j) = Hz.col(2);

            Mz_dl.col(j) = std::complex<double>(0.0,1.0)*Hz.col(0)*dt/2.0;
            Mz_d.col(j)  = 1+std::complex<double>(0.0,1.0)*Hz.col(1)*dt/2.0;
            Mz_du.col(j) = std::complex<double>(0.0,1.0)*Hz.col(2)*dt/2.0;

            Mpz_dl.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(0)*dt/2.0;
            Mpz_d.col(j)  = 1-std::complex<double>(0.0,1.0)*Hz.col(1)*dt/2.0;
            Mpz_du.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(2)*dt/2.0;
        }
        
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,p.Nx,p.Nz);
        StepX(Mx_dl,Mx_d,Mx_du,Mpx_dl,Mpx_d,Mpx_du,PsiZ,Psi,p.Nx,p.Nz);
        Psi = Psi%Mask;
        accX(i) = AcceX(Psi,V,VecPotentialX(i),x,z);
        accZ(i) = AcceZ(Psi,V,VecPotentialZ(i),x,z);
        if(i%p.Nsteps==0){
            energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
            norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
            normVec(i%p.Nsteps) = norm;
            enerVec(i%p.Nsteps) = energy;
            #ifdef TEXTOUTPUT
                outfile<<"Step: "<<i<<" Energy: "<<energy<<"Norm: "<<norm<<std::endl;
            #else
                std::cout<<"Step: "<<i<<" Energy: "<<energy<<"Norm: "<<norm<<std::endl;
            #endif
        }
    }
    accX = accX%accMaskX;
    accZ = accZ%accMaskZ;
    accX.save("results/accX.dat",arma::raw_ascii);
    accZ.save("results/accZ.dat",arma::raw_ascii);
    ElectricFieldX.save("results/ElectricFieldX.dat",arma::raw_ascii);
    ElectricFieldZ.save("results/ElectricFieldZ.dat",arma::raw_ascii);
    MagneticFieldX.save("results/MagneticFieldX.dat",arma::raw_ascii);
    MagneticFieldZ.save("results/MagneticFieldZ.dat",arma::raw_ascii);
    VecPotentialX.save("results/VecPotentialX.dat",arma::raw_ascii);
    VecPotentialZ.save("results/VecPotentialZ.dat",arma::raw_ascii);
    Psi.save("results/Psi.dat",arma::raw_ascii);
       
    return 0;
}

