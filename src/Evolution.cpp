#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include <omp.h>
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
    outfile.open("output_Evolution.txt");
    #endif
/*
    double rmin = p.rmin;
    double rmax = p.rmax;
    int Nr = p.Nr;
    double dr = p.dr; 
    double zmin = p.zmin;
    double zmax = p.zmax;
    int Nz = p.Nz;
    double dz = p.dz;

    double w0E = p.w0E;
    double t0 = p.t0;
    double tmax = p.tmax;
    double dt = p.dt;
    int Nt = p.Nt;
    int Nsteps = p.Nsteps;
*/    
    #ifdef TEXTOUTPUT
    outfile<<"Parameters:\n";
    outfile<<"-------------\n";
    outfile<<"rmin: "<<p.rmin<<std::endl;
    outfile<<"rmax: "<<p.rmax<<std::endl;
    outfile<<"Nr: "<<p.Nr<<std::endl;
    outfile<<"dr: "<<p.dr<<std::endl;
    outfile<<"zmin: "<<p.zmin<<std::endl;
    outfile<<"zmax: "<<p.zmax<<std::endl;
    outfile<<"Nz: "<<p.Nz<<std::endl;
    outfile<<"dz: "<<p.dz<<std::endl;
    outfile<<"tlim: "<<p.tmax<<std::endl;
    outfile<<"Nt: "<<p.Nt<<std::endl;
    outfile<<"dt: "<<p.dt<<std::endl;
    #else
    std::cout<<"Parameters:\n";
    std::cout<<"-------------\n";
    std::cout<<"rmin: "<<p.rmin<<std::endl;
    std::cout<<"rmax: "<<p.rmax<<std::endl;
    std::cout<<"Nr: "<<p.Nr<<std::endl;
    std::cout<<"dr: "<<p.dr<<std::endl;
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

    arma::dmat t = arma::linspace(p.t0,p.tmax,p.Nt);
    arma::dmat ElectricField = arma::colvec(p.Nt,arma::fill::zeros);
    arma::dmat MagneticField = arma::colvec(p.Nt,arma::fill::zeros);
    arma::dmat VecPotential = arma::colvec(p.Nt,arma::fill::zeros);
    arma::vec r = arma::linspace(p.rmin,p.rmax,p.Nr);
    arma::vec z = arma::linspace(p.zmin,p.zmax,p.Nz);
    arma::mat V(p.Nr,p.Nz,arma::fill::zeros);
    arma::mat dV(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Psi(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiOld(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiR(p.Nr,p.Nz,arma::fill::zeros);
    arma::dmat Psi2(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat PsiZ(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat MaskZ(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat MaskR(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mask(p.Nr,p.Nz,arma::fill::zeros);
    arma::dmat R(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_colvec acc(p.Nt,arma::fill::zeros);
    arma::cx_colvec accMask(p.Nt,arma::fill::zeros);
    arma::cx_colvec normVec(p.Nt/p.Nsteps,arma::fill::zeros);
    arma::cx_colvec enerVec(p.Nt/p.Nsteps,arma::fill::zeros);

    arma::cx_mat Hr_dl(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Hr_d(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Hr_du(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mr_dl(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mr_d(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mr_du(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpr_dl(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpr_d(p.Nr,p.Nz,arma::fill::zeros);
    arma::cx_mat Mpr_du(p.Nr,p.Nz,arma::fill::zeros);

    arma::cx_mat Hz_dl(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Hz_d(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Hz_du(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mz_dl(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mz_d(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mz_du(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mpz_dl(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mpz_d(p.Nz,p.Nr,arma::fill::zeros);
    arma::cx_mat Mpz_du(p.Nz,p.Nr,arma::fill::zeros);
    

    r = r+p.dr/2.0;
    for(int i = 0; i<p.Nr;i++){
        R.row(i) = r(i)*arma::ones<arma::rowvec>(p.Nz);
    }
    //R.save("results/R.dat",arma::raw_ascii);
    //r.save("results/r.dat",arma::raw_ascii);
    //z.save("results/z.dat",arma::raw_ascii);


    for(int i=0; i<p.Nt;i++){
        MagneticField(i) = BField(t(i),p);
        ElectricField(i) = EField(t(i),p);
    }

    for(int i=0; i<p.Nt;i++){
        VecPotential(i) = -137.04*intSimpson(EField,0,t(i),6000,p);
    }

    CoulombPotential(V,r,z);
    //V.save("results/Coulomb.dat",arma::raw_ascii);
    Psi.load("results/PsiGround.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    //Psi2.save("results/PsiProb.dat",arma::raw_ascii);
    derivativeZ(V,z,dV);
    //dV.save("results/dV.dat",arma::raw_ascii);
    maskZ(MaskZ,r,z,12.0,1.0);
    maskR(MaskR,r,z,10.0,1.0);
    accelerationMask(accMask, t, p);
    accMask.save("results/accMask.dat",arma::raw_ascii);
    //MaskZ.save("results/MaskZ.dat",arma::raw_ascii);
    //MaskR.save("results/MaskR.dat",arma::raw_ascii);
    norm = 2.0*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*p.dr,0)*p.dz,1));
    
    #ifdef TEXTOUTPUT
    outfile<<"Norm: "<<norm<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
    #else
    std::cout<<"Norm: "<<norm<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
    #endif
    Mask = MaskZ%MaskR;
    double start = omp_get_wtime();
    for(int i=0;i<p.Nt;i++){
	
        double start_step = omp_get_wtime();

        #pragma omp parallel for
        for (int j=0; j<z.n_elem;j++){
            arma::cx_mat Hr(p.Nr,3,arma::fill::zeros);
            HamR(Hr,V, MagneticField(i),r,p.dr,R,j);
            Hr_dl.col(j) = Hr.col(0);
            Hr_d.col(j) = Hr.col(1);
            Hr_du.col(j) = Hr.col(2);
            Mr_dl.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(0)*p.dt/2.0;
            Mr_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hr.col(1)*p.dt/2.0;
            Mr_du.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(2)*p.dt/2.0;
            Mpr_dl.col(j) = std::complex<double>(0.0,1.0)*Hr.col(0)*p.dt/2.0;
            Mpr_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hr.col(1)*p.dt/2.0;
            Mpr_du.col(j) = std::complex<double>(0.0,1.0)*Hr.col(2)*p.dt/2.0;

            }
        #pragma omp parallel for
        for (int j=0; j<r.n_elem;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,VecPotential(i),MagneticField(i),z,p.dz,R,j);
            Hz_dl.col(j) = Hz.col(0);
            Hz_d.col(j) = Hz.col(1);
            Hz_du.col(j) = Hz.col(2);
            Mz_dl.col(j) = std::complex<double>(0.0,1.0)*Hz.col(0)*p.dt/4.0;
            Mz_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hz.col(1)*p.dt/4.0;
            Mz_du.col(j) = std::complex<double>(0.0,1.0)*Hz.col(2)*p.dt/4.0;
            Mpz_dl.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(0)*p.dt/4.0;
            Mpz_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hz.col(1)*p.dt/4.0;
            Mpz_du.col(j) = -std::complex<double>(0.0,1.0)*Hz.col(2)*p.dt/4.0;
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
    
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,p.Nr,p.Nz);       
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
        StepR(Mr_dl,Mr_d,Mr_du,Mpr_dl,Mpr_d,Mpr_du,PsiZ,PsiR,p.Nr,p.Nz);
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
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiR,Psi,p.Nr,p.Nz);
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

	    #ifdef DEBUG2
           double start_acc = omp_get_wtime();
	    #endif
        acc(i) = AcceZ(Psi,V,VecPotential(i),MagneticField(i),R,r,z);
	    #ifdef DEBUG2
  	        double end_acc = omp_get_wtime();
	        #ifdef TEXTOUTPUT
	        outfile<<"[DEBUG2] Acc exectime: "<<(end_acc-start_acc)*1000<<" ms\n";
            #else
	        std::cout<<"[DEBUG2] Acc exectime: "<<(end_acc-start_acc)*1000<<" ms\n";
            #endif
    	#endif

        //PsiOld = PsiR2/sqrt(Norm);
	    double end_step = omp_get_wtime();
	    if (i%p.Nsteps==0){
            norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*p.dr,0)*p.dz,1));
            energy = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);
            normVec(i%p.Nsteps) = norm;
            enerVec(i%p.Nsteps) = energy;
 	    
            #ifdef TEXTOUTPUT
	            outfile<<"[DEBUG] Time from init: "<<(end_step-start)<<"\n";
          	    outfile<<"Step: "<<i<<" Norm: "<<norm<<" Energy "<< energy<<"\n\n";
            #else
	            std::cout<<"[DEBUG] Time from init: "<<(end_step-start)<<"\n";
          	    std::cout<<"Step: "<<i<<" Norm: "<<norm<<" Energy "<< energy<<"\n\n";
            #endif
        }
    }
    acc = acc%accMask;
    double end = omp_get_wtime();
    
    #ifdef TEXTOUTPUT 
    outfile <<"Simulation exectime: "<<(end-start)*1000<<std::endl;
    outfile <<"Timestep exectime: "<<(end-start)*1000/p.Nt<<std::endl;
    #else
    std::cout <<"Simulation exectime: "<<(end-start)*1000<<std::endl;
    std::cout <<"Timestep exectime: "<<(end-start)*1000/p.Nt<<std::endl;
    #endif

    Psi2 = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("results/PsiEnd.dat",arma::raw_ascii);
    acc.save("results/acc.dat",arma::raw_ascii);
    //normVec.save("results/normVec.dat",arma::raw_ascii);
    //enerVec.save("results/enerVer.dat",arma::raw_ascii);
    MagneticField.save("results/MagneticField.dat",arma::raw_ascii);
    VecPotential.save("results/VecPotential.dat",arma::raw_ascii);
    ElectricField.save("results/ElectricField.dat",arma::raw_ascii);
    //PsiOld.save("results/PsiGround.dat",arma::raw_ascii);

    #ifdef TEXTOUTPUT
    outfile.close();
    #endif
    return 0;
}
