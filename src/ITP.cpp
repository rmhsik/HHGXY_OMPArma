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

#define OMP_NUM_THREADS 4
int main(){
    omp_set_num_threads(OMP_NUM_THREADS);
    parameters p;

    #ifdef TEXTOUTPUT
    std::ofstream outfile;
    outfile.open("output.txt");
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
    outfile<<"Nt: "<<p.Nt_ITP<<std::endl;
    outfile<<"dt: "<<p.dt_ITP<<std::endl;
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
    std::cout<<"Nt: "<<p.Nt_ITP<<std::endl;
    std::cout<<"dt: "<<p.dt_ITP<<std::endl;
    #endif
    std::complex<double> norm;
    std::complex<double> energy;

    arma::dmat t = arma::linspace(p.t0,p.tmax,p.Nt_ITP);
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
    //R.save("R.dat",arma::raw_ascii);
    //r.save("r.dat",arma::raw_ascii);
    //z.save("z.dat",arma::raw_ascii);


    CoulombPotential(V,r,z);
    Exponential(Psi,r,z,0.0,0.0,1.0);
    norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*p.dr,0)*p.dz,1));
    //std::complex<double> rExpected;
    std::cout<< norm <<std::endl;
    Psi = Psi/sqrt(norm);
    std::cout<< 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*p.dr,0)*p.dz,1))<<std::endl;

    //V.save("results/Coulomb.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("results/PsiProb.dat",arma::raw_ascii);

    std::cout<<"Mask\n";
    maskZ(MaskZ,r,z,12.0,1.0);
    maskR(MaskR,r,z,p.rmax*0.1,1.0);
    Mask = MaskR%MaskZ;
    //MaskZ.save("results/MaskZ.dat",arma::raw_ascii);
    //MaskR.save("results/MaskR.dat",arma::raw_ascii);
    energy = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);
    std::cout<<energy<<std::endl;
    for(int i=0;i<p.Nt_ITP;i++){
        double start_step = omp_get_wtime();
        std::complex<double> dt = std::complex<double> (0.0,-1.0)*p.dt_ITP;
        #pragma omp parallel for
        for (int j=0; j<z.n_elem;j++){
            arma::cx_mat Hr(p.Nr,3,arma::fill::zeros);
            HamR(Hr,V, 0.0,r,p.dr,R,j);
            Hr_dl.col(j) = Hr.col(0);
            Hr_d.col(j) = Hr.col(1);
            Hr_du.col(j) = Hr.col(2);
            Mr_dl.col(j) = std::complex<double>(0.0,1.0)*Hr.col(0)*dt/2.0;
            Mr_d.col(j) = 1.0 + std::complex<double>(0.0,1.0)*Hr.col(1)*dt/2.0;
            Mr_du.col(j) = std::complex<double>(0.0,1.0)*Hr.col(2)*dt/2.0;
            Mpr_dl.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(0)*dt/2.0;
            Mpr_d.col(j) = 1.0 - std::complex<double>(0.0,1.0)*Hr.col(1)*dt/2.0;
            Mpr_du.col(j) = -std::complex<double>(0.0,1.0)*Hr.col(2)*dt/2.0;

            }
        #pragma omp parallel for
        for (int j=0; j<r.n_elem;j++){
            arma::cx_mat Hz(p.Nz,3,arma::fill::zeros);
            HamZ(Hz,V,0.0,0.0,z,p.dz,R,j);
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

 
        norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*p.dr,0)*p.dz,1));
        energy = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);

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
