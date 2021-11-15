#include <iostream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include "physics.h"
#include "fields.h"
#include "math_aux.h"
#include "param.h"
#define ARMA_NO_DEBUG

int main(){
    omp_set_num_threads(OMP_NUM_THREADS);
    parameters p;

    #ifdef TEXTOUTPUT
    std::ofstream outfile;
    outfile.open("output_ITP.txt");
    #endif
/*
    double xmin = p.xmin;
    double xmax = p.xmax;
    int Nx = p.Nx;
    double dx = p.dx; 
    double zmin = p.zmin;
    double zmax = p.zmax;
    int Nz = p.Nz;
    double dz = p.dz;

    double w0Ex = p.w0Ex;
    double w0Ez = p.w0Ez;
    double t0 = p.t0;
    double tmax = p.tmax;
    double dt = p.dt;
    int Nt = p.Nt;
    int Nsteps = p.Nsteps;
*/    
    #ifdef TEXTOUTPUT
    outfile<<"Parameters:\n";
    outfile<<"-------------\n";
    outfile<<"xmin: "<<p.xmin<<std::endl;
    outfile<<"xmax: "<<p.xmax<<std::endl;
    outfile<<"Nx: "<<p.Nx<<std::endl;
    outfile<<"dx: "<<p.dx<<std::endl;
    outfile<<"zmin: "<<p.zmin<<std::endl;
    outfile<<"zmax: "<<p.zmax<<std::endl;
    outfile<<"Nz: "<<p.Nz<<std::endl;
    outfile<<"dz: "<<p.dz<<std::endl;
    outfile<<"tlim: "<<p.tmax<<std::endl;
    outfile<<"Nt: "<<p.Nt_ITP<<std::endl;
    outfile<<"dt: "<<p.dt_ITP<<std::endl;
    #else
    std::cout<<"Parameters:\n";
    std::cout<<"-------------\n";
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
    #endif
    std::complex<double> norm;
    std::complex<double> energy;

    arma::dmat t = arma::linspace(p.t0,p.tmax,p.Nt_ITP);
    //arma::vec x = arma::linspace(p.xmin,p.xmax,p.Nx);
    arma::vec x(1,arma::fill::zeros);
    arma::vec z = arma::linspace(p.zmin,p.zmax,p.Nz);
    arma::mat V(p.Nx,p.Nz,arma::fill::zeros);
    arma::mat dV(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Psi(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiOld(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiX(p.Nx,p.Nz,arma::fill::zeros);
    arma::dmat Psi2(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiZ(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat MaskZ(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat MaskX(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mask(p.Nx,p.Nz,arma::fill::zeros);
    //arma::dmat R(p.Nr,p.Nz,arma::fill::zeros);

    arma::cx_mat Hx_dl(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Hx_d(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Hx_du(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mx_dl(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mx_d(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mx_du(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpx_dl(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpx_d(p.Nx,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpx_du(p.Nx,p.Nz,arma::fill::zeros);

    arma::cx_mat Hz_dl(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Hz_d(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Hz_du(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mz_dl(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mz_d(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mz_du(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mpz_dl(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mpz_d(p.Nz,p.Nx,arma::fill::zeros);
    arma::cx_mat Mpz_du(p.Nz,p.Nx,arma::fill::zeros);

    
    //r = r+p.dr/2.0;
    //for(int i = 0; i<p.Nr;i++){
    //    R.row(i) = r(i)*arma::ones<arma::rowvec>(p.Nz);
    //}
    //R.save("R.dat",arma::raw_ascii);
    //r.save("r.dat",arma::raw_ascii);
    //z.save("z.dat",arma::raw_ascii);


    CoulombPotential(V,x,z);
    Gaussian(Psi,x,z,0.0,0.0,1.0);
    norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
    //std::complex<double> rExpected;
    std::cout<< norm <<std::endl;
    Psi = Psi/sqrt(norm);
    std::cout<< arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1))<<std::endl;

    //V.save("results/Coulomb.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("results/PsiProb.dat",arma::raw_ascii);

    std::cout<<"Mask\n";
    maskZ(MaskZ,x,z,12.0,1.0);
    //maskX(MaskX,x,z,12.0,1.0);
    Mask = MaskZ;
    //MaskZ.save("results/MaskZ.dat",arma::raw_ascii);
    //MaskR.save("results/MaskR.dat",arma::raw_ascii);
    energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);
    std::cout<<energy<<std::endl;
    for(int i=0;i<p.Nt_ITP;i++){
        double start_step = omp_get_wtime();
        std::complex<double> dt = std::complex<double> (0.0,-1.0)*p.dt_ITP;
        /*#pragma omp parallel for
        for (int j=0; j<z.n_elem;j++){
            arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
            HamX(Hx,V, 0.0,x,p.dx,j);
            Hx_dl.col(j) = Hx.col(0);
            Hx_d.col(j) = Hx.col(1);
            Hx_du.col(j) = Hx.col(2);
            Mx_dl.col(j) = std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mx_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mx_du.col(j) = std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;
            Mpx_dl.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mpx_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mpx_du.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;

            }*/
        #pragma omp parallel for
        for (int j=0; j<x.n_elem;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,0.0,z,p.dz,j);
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
        #ifdef DEBUG2
            double end_matrices = omp_get_wtime();
            #ifdef TEXTOUTPUT
            outfile<<"[DEBUG2] Setup matrices exectime: "<<(end_matrices - start_step)*1000<< " ms\n";
            #else
            std::cout<<"[DEBUG2] Setup matrices exectime: "<<(end_matrices - start_step)*1000<< " ms\n";
            #endif
        #endif
	    #ifdef DEBUG2
            double start_stepz1 = omp_get_wtime();
        #endif 
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,p.Nx,p.Nz);       
        #ifdef DEBUG2
           double end_stepz1 = omp_get_wtime();
           #ifdef TEXTOUTPUT
	       outfile<<"[DEBUG2] StepZ_1 exectime: "<<(end_stepz1-start_stepz1)*1000<<" ms\n";
           #else
	       std::cout<<"[DEBUG2] StepZ_1 exectime: "<<(end_stepz1-start_stepz1)*1000<<" ms\n";
           #endif
        #endif         
	
 	    #ifdef DEBUG2
	        double start_stepr = omp_get_wtime();
	    #endif
        //StepX(Mx_dl,Mx_d,Mx_du,Mpx_dl,Mpx_d,Mpx_du,PsiZ,PsiX,p.Nx,p.Nz);
        #ifdef DEBUG2
           double end_stepr = omp_get_wtime();
           #ifdef TEXTOUTPUT
           outfile<<"[DEBUG2] StepR exectime: "<<(end_stepr-start_stepr)*1000<<" ms\n";
           #else
           std::cout<<"[DEBUG2] StepR exectime: "<<(end_stepr-start_stepr)*1000<<" ms\n";
           #endif
        #endif
       
        #ifdef DEBUG2
           double start_stepz2 = omp_get_wtime();
	    #endif
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiZ,Psi,p.Nx,p.Nz);
        #ifdef DEBUG2
           double end_stepz2 = omp_get_wtime();
           #ifdef TEXTOUTPUT
	       outfile<<"[DEBUG2] StepZ_2 exectime: "<<(end_stepz2-start_stepz2)*1000<<" ms\n";
           #else
    	   std::cout<<"[DEBUG2] StepZ_2 exectime: "<<(end_stepz2-start_stepz2)*1000<<" ms\n";
           #endif
        #endif 

	    #ifdef DEBUG2
           double start_mask = omp_get_wtime();
        #endif
        Psi = Psi%Mask;
        #ifdef DEBUG2
           double end_mask = omp_get_wtime();
           #ifdef TEXTOUTPUT 
	       outfile<<"[DEBUG2] Mask execitme: "<<(end_mask-start_mask)*1000<<" ms\n";
           #else
	       std::cout<<"[DEBUG2] Mask execitme: "<<(end_mask-start_mask)*1000<<" ms\n";
           #endif
        #endif

 
        norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi,0)*p.dz,1));
        energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);

        Psi = Psi/sqrt(norm);
        std::cout<<i<<": ";
        std::cout<<energy<<" "<<norm<<std::endl;
    }

    std::cout<<"Standard\n";
    for(int i=0;i<p.Nt_ITP;i++){
        double start_step = omp_get_wtime();
        std::complex<double> dt = std::complex<double> (1.0,0.0)*p.dt_ITP;
        #pragma omp parallel for
        for (int j=0; j<z.n_elem;j++){
            arma::cx_mat Hx(p.Nx,3,arma::fill::zeros);
            HamX(Hx,V, 0.0,x,p.dx,j);
            Hx_dl.col(j) = Hx.col(0);
            Hx_d.col(j) = Hx.col(1);
            Hx_du.col(j) = Hx.col(2);
            Mx_dl.col(j) = std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mx_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mx_du.col(j) = std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;
            Mpx_dl.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(0)*dt/2.0;
            Mpx_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hx.col(1)*dt/2.0;
            Mpx_du.col(j) = -std::complex<double>(0.0,1.0)*Hx.col(2)*dt/2.0;

            }
        #pragma omp parallel for
        for (int j=0; j<x.n_elem;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,0.0,z,p.dz,j);
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
        #ifdef DEBUG2
            double end_matrices = omp_get_wtime();
            #ifdef TEXTOUTPUT
            outfile<<"[DEBUG2] Setup matrices exectime: "<<(end_matrices - start_step)*1000<< " ms\n";
            #else
            std::cout<<"[DEBUG2] Setup matrices exectime: "<<(end_matrices - start_step)*1000<< " ms\n";
            #endif
        #endif
	    #ifdef DEBUG2
            double start_stepz1 = omp_get_wtime();
        #endif 
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,p.Nx,p.Nz);       
        #ifdef DEBUG2
           double end_stepz1 = omp_get_wtime();
           #ifdef TEXTOUTPUT
	       outfile<<"[DEBUG2] StepZ_1 exectime: "<<(end_stepz1-start_stepz1)*1000<<" ms\n";
           #else
	       std::cout<<"[DEBUG2] StepZ_1 exectime: "<<(end_stepz1-start_stepz1)*1000<<" ms\n";
           #endif
        #endif         
	
 	    #ifdef DEBUG2
	        double start_stepr = omp_get_wtime();
	    #endif
        StepX(Mx_dl,Mx_d,Mx_du,Mpx_dl,Mpx_d,Mpx_du,PsiZ,PsiX,p.Nx,p.Nz);
        #ifdef DEBUG2
           double end_stepr = omp_get_wtime();
           #ifdef TEXTOUTPUT
           outfile<<"[DEBUG2] StepR exectime: "<<(end_stepr-start_stepr)*1000<<" ms\n";
           #else
           std::cout<<"[DEBUG2] StepR exectime: "<<(end_stepr-start_stepr)*1000<<" ms\n";
           #endif
        #endif
       
        #ifdef DEBUG2
           double start_stepz2 = omp_get_wtime();
	    #endif
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiX,Psi,p.Nx,p.Nz);
        #ifdef DEBUG2
           double end_stepz2 = omp_get_wtime();
           #ifdef TEXTOUTPUT
	       outfile<<"[DEBUG2] StepZ_2 exectime: "<<(end_stepz2-start_stepz2)*1000<<" ms\n";
           #else
    	   std::cout<<"[DEBUG2] StepZ_2 exectime: "<<(end_stepz2-start_stepz2)*1000<<" ms\n";
           #endif
        #endif 

	    #ifdef DEBUG2
           double start_mask = omp_get_wtime();
        #endif
        Psi = Psi%Mask;
        #ifdef DEBUG2
           double end_mask = omp_get_wtime();
           #ifdef TEXTOUTPUT 
	       outfile<<"[DEBUG2] Mask execitme: "<<(end_mask-start_mask)*1000<<" ms\n";
           #else
	       std::cout<<"[DEBUG2] Mask execitme: "<<(end_mask-start_mask)*1000<<" ms\n";
           #endif
        #endif

 
        norm = arma::as_scalar(arma::sum(arma::sum(arma::conj(Psi)%Psi*p.dx,0)*p.dz,1));
        energy = Energy(Hx_dl,Hx_d,Hx_du,Hz_dl,Hz_d,Hz_du,Psi,x,z);

        Psi = Psi/sqrt(norm);
        std::cout<<i<<": ";
        std::cout<<energy<<" "<<norm<<std::endl;
    }




    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("results/PsiGround2.dat",arma::raw_ascii);
    Psi.save("results/PsiGround.dat",arma::raw_ascii);
    //Hr.save("Hr.dat",arma::raw_ascii);
    //Hz.save("Hz.dat",arma::raw_ascii);
    return 0;
}
