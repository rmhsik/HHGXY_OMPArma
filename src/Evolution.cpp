#include <iostream>
#include <fstream>
#include <cmath>
#include <complex>
#include <armadillo>
#include <typeinfo>
#include <omp.h>
#include "../include/physics.h"
#include "../include/fields.h"
#include "../include/math_aux.h"
#define ARMA_NO_DEBUG

#define OMP_NUM_THREADS 4

int main(){
    omp_set_num_threads(OMP_NUM_THREADS);

    #ifdef TEXTOUTPUT
    std::ofstream outfile;
    outfile.open("output.txt");
    #endif

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
    int Nsteps = 50;
    
    #ifdef TEXTOUTPUT
    outfile<<"Parameters:\n";
    outfile<<"-------------\n";
    outfile<<"rmin: "<<rmin<<std::endl;
    outfile<<"rmax: "<<rmax<<std::endl;
    outfile<<"Nr: "<<Nr<<std::endl;
    outfile<<"dr: "<<dr<<std::endl;
    outfile<<"zmin: "<<zmin<<std::endl;
    outfile<<"zmax: "<<zmax<<std::endl;
    outfile<<"Nz: "<<Nz<<std::endl;
    outfile<<"dz: "<<dz<<std::endl;
    outfile<<"tlim: "<<tmax<<std::endl;
    outfile<<"Nt: "<<Nt<<std::endl;
    outfile<<"dt: "<<dt<<std::endl;
    #else
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
    #endif
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
    Psi.load("PsiGround_120_120.dat",arma::raw_ascii);

    //V.save("Coulomb.dat",arma::raw_ascii);
    Psi2  = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    //Psi2.save("PsiProb.dat",arma::raw_ascii);

    //PsiOld = Psi;

    derivativeZ(V,z,dV);
    //dV.save("dV.dat",arma::raw_ascii);

    maskZ(MaskZ,r,z,12.0,1.0);
    maskR(MaskR,r,z,10.0,1.0);
    //MaskZ.save("MaskZ.dat",arma::raw_ascii);
    //MaskR.save("MaskR.dat",arma::raw_ascii);
    norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
    
    #ifdef TEXTOUTPUT
    outfile<<"Norm: "<<norm<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
    #else
    std::cout<<"Norm: "<<norm<<" Energy: "<<Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z)<<std::endl;
    #endif
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
    
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,Psi,PsiZ,Nr,Nz);       
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
        StepR(Mr_dl,Mr_d,Mr_du,Mpr_dl,Mpr_d,Mpr_du,PsiZ,PsiR,Nr,Nz);
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
        StepZ(Mz_dl,Mz_d,Mz_du,Mpz_dl,Mpz_d,Mpz_du,PsiR,Psi,Nr,Nz);
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
	    if (i%Nsteps==0){
            norm = 2*M_PI*arma::as_scalar(arma::sum(arma::sum(R%arma::conj(Psi)%Psi*dr,0)*dz,1));
            energy = Energy(Hr_dl,Hr_d,Hr_du,Hz_dl,Hz_d,Hz_du,Psi,R,r,z);
            normVec(i%Nsteps) = norm;
            enerVec(i%Nsteps) = energy;
 	    
            #ifdef TEXTOUTPUT
	            outfile<<"[DEBUG] Time from init: "<<(end_step-start)<<"\n";
          	    outfile<<"Step: "<<i<<" Norm: "<<norm<<" Energy "<< energy<<"\n\n";
            #else
	            std::cout<<"[DEBUG] Time from init: "<<(end_step-start)<<"\n";
          	    std::cout<<"Step: "<<i<<" Norm: "<<norm<<" Energy "<< energy<<"\n\n";
            #endif
        }
    }

    double end = omp_get_wtime();
    
    #ifdef TEXTOUTPUT 
    outfile <<"Simulation exectime: "<<(end-start)*1000<<std::endl;
    outfile <<"Timestep exectime: "<<(end-start)*1000/Nt<<std::endl;
    #else
    std::cout <<"Simulation exectime: "<<(end-start)*1000<<std::endl;
    std::cout <<"Timestep exectime: "<<(end-start)*1000/Nt<<std::endl;
    #endif

    Psi2 = arma::conv_to<arma::dmat>::from(arma::conj(Psi)%Psi);
    Psi2.save("PsiEnd1.dat",arma::raw_ascii);
    acc.save("acc1.dat",arma::raw_ascii);
    //normVec.save("normVec1.dat",arma::raw_ascii);
    //enerVec.save("enerVer1.dat",arma::raw_ascii);
    MagneticField.save("MagneticField1.dat",arma::raw_ascii);
    VecPotential.save("VecPotential1.dat",arma::raw_ascii);
    ElectricField.save("ElectricField1.dat",arma::raw_ascii);
    //PsiOld.save("PsiGround.dat",arma::raw_ascii);

    #ifdef TEXTOUTPUT
    outfile.close();
    #endif
    return 0;
}
