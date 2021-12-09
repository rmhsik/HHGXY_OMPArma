#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include "physics.h"
#include "math_aux.h"
#include "param.h"
#define ARMA_NO_DEBUG

int main(){
    parameters p;

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
    std::cout<<"Nt: "<<p.Nt_ITP<<std::endl;
    std::cout<<"dt: "<<p.dt_ITP<<std::endl;

    std::complex<double> norm;
    std::complex<double> energy;

    arma::vec t = arma::linspace(p.t0,p.tmax, p.Nt_ITP);
    arma::vec x = arma::linspace(p.xmin,p.xmax,p.Nx);
    arma::vec z = arma::linspace(p.zmin,p.zmax,p.Nz);
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
    std::cout<<"Matrices definition\n";
    
    CoulombPotential(V,x,z);
    Gaussian(Psi, x,z,0.0, 0.0, 3.0);

    std::cout<<"Coulomb, Gaussian\n";

    norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
    std::cout<<norm<<std::endl;
    Psi = Psi/sqrt(norm);
    norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
    std::cout<<norm<<std::endl;
    std::cout<<"Norm\n";
    
    maskZ(MaskZ,x,z,12.0,1.0);
    maskX(MaskX,x,z,12.0,1.0);
    Mask = MaskZ%MaskX;
    std::cout<<"Mask\n";

    arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
    arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
    for(int i=0;i<p.Nx;i++){
        HamZ(Hz,V,0.0,0.0,x,z,p.dx,p.dz,i);
        Hz_dl.col(i) = Hz.col(0);
        Hz_d.col(i) = Hz.col(1);
        Hz_du.col(i) = Hz.col(2);
    }
    std::cout<<"HamZ\n";
    for(int i=0;i<p.Nz;i++){
        HamX(Hx,V,0.0,0.0,x,z,p.dx,p.dz,i);
        Hx_dl.col(i) = Hx.col(0);
        Hx_d.col(i) = Hx.col(1);
        Hx_du.col(i) = Hx.col(2);
    }
    std::cout<<"HamX\n";
   
    energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
    std::cout<<"Energy: "<<energy<<std::endl;

    std::complex<double> dt =std::complex<double>(0.0,-1.0)*p.dt_ITP;
    for (int i=0; i<p.Nt_ITP;i++){
        for(int j=0;j<p.Nz;j++){
            arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
            HamX(Hx,V,0.0,0.0,x,z,p.dx,p.dz,j);
            Hx_dl.col(j) = Hx.col(0);
            Hx_d.col(j) = Hx.col(1);
            Hx_du.col(j) = Hx.col(2);

            Mx_dl.col(j) = std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mx_d.col(j)  = 1+std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mx_du.col(j) = std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;

            Mpx_dl.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mpx_d.col(j)  = 1-std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mpx_du.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;
        }
        for(int j=0;j<p.Nx;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,0.0,0.0,x,z,p.dx,p.dz,j);
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
        StepX(Mx_dl,Mx_d,Mx_du,Mpx_dl,Mpx_d,Mpx_du,Psi,PsiX,p.Nx,p.Nz);
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiX,PsiZ,p.Nx,p.Nz);
        Psi = PsiZ;
        energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
        norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
        Psi = Psi/sqrt(norm);
        std::cout<<"Energy: "<<energy<<"Norm: "<<norm<<std::endl;
    }
    
    Psi.save("results/PsiGround.dat",arma::raw_ascii);
    std::cout<<" Standard!!"<<std::endl;
    dt = std::complex<double>(1.0,0.0)*p.dt_ITP;
    for (int i=0; i<p.Nt_ITP;i++){
        for(int j=0;j<p.Nz;j++){
            arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
            HamX(Hx,V,0.0,0.0,x,z,p.dx,p.dz,j);
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
            HamZ(Hz,V,0.0,0.0,x,z,p.dx,p.dz,j);
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
        StepX(Mx_dl,Mx_d,Mx_du,Mpx_dl,Mpx_d,Mpx_du,Psi,PsiX,p.Nx,p.Nz);
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiX,PsiZ,p.Nx,p.Nz);
        Psi = PsiZ;
        energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
        norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
        //Psi = Psi/sqrt(norm);
        std::cout<<"Energy: "<<energy<<"Norm: "<<norm<<std::endl;
    }
    
    return 0;
}

