#ifndef PARAM_H
#define PARAM_H
#include <cmath>
#include <iostream>

#define OMP_NUM_THREADS 4
const double c = 137.04;

struct parameters{
        
	// GRID
        double xmin     = -120.0;
        double xmax     = 120.0;
        int Nx          = 240;
        double dx       = (xmax-xmin)/Nx;
        double zmin     = -120.0;
        double zmax     =  120.0;
        int Nz          = 240; 
        double dz       = (zmax-zmin)/Nz;
        
  	// ELECTROMAGNETIC FIELDS
        double w0Ex     = 0.057;
        double w0Ez     = 0.057;
        double w0Bx     = 0.057;
        double w0Bz     = 0.057;
        double phiEx    = 0.0;
        double phiEz    = 0.0;
        double phiBx    = 0.0*M_PI;
        double phiBz    = 0.0;
        double E0x      = 0.067;
        double E0z      = 0.0;
        double B0x      = 0.0;
        double B0z      = 0.12;
        int env         = 0; //0 -> sin2; 1 -> trap 
        
	// TIME (FIELD AND SIMULATION)
        double t0       = 0.0;
        double dt       = 0.02;
        double simPeriods   = 4.0;
        double fieldPeriods = 4.0;
        int Nsteps      = 50;
    	double tmax;	
        int Nt;
        double dt_ITP   = 0.01;
        int Nt_ITP      = 1000;
        
	//ACCELERATION MASK
	double fwhm_accMask = 0.001;
        
	parameters(){
            if(w0Bz<=w0Ez){
                tmax = simPeriods*2.0*M_PI/w0Bz;
            }
            else{ 
                tmax = simPeriods*2.0*M_PI/w0Ez;
            }
            Nt = (tmax-t0)/dt;
        }
        void printParameters(){
            std::cout<<"Parameters:\n";
            std::cout<<"-----------\n";
            std::cout<<"\t xmin: "<<xmin<<std::endl;
            std::cout<<"\t xmax: "<<xmax<<std::endl;
            std::cout<<"\t Nx: "<<Nx<<std::endl;
            std::cout<<"\t dx: "<<dx<<std::endl;
            std::cout<<"\t zmin: "<<zmin<<std::endl;
            std::cout<<"\t zmax: "<<zmax<<std::endl;
            std::cout<<"\t Nz: "<<Nz<<std::endl;
            std::cout<<"\t dz: "<<dz<<std::endl;
            std::cout<<"\t w0Ex: "<<w0Ex<<std::endl;
            std::cout<<"\t w0Ez: "<<w0Ez<<std::endl;
            std::cout<<"\t w0Bx: "<<w0Bx<<std::endl;
            std::cout<<"\t w0Bz: "<<w0Bz<<std::endl;
            std::cout<<"\t t0: "<<t0<<std::endl;
            std::cout<<"\t tmax: "<<tmax<<std::endl;
            std::cout<<"\t dt: "<<dt<<std::endl;           
            std::cout<<"\t Nt: "<<Nt<<std::endl;
            std::cout<<"\t phiEx: "<<phiEx<<std::endl;
            std::cout<<"\t phiEz: "<<phiEz<<std::endl;
            std::cout<<"\t phiBx: "<<phiBx<<std::endl;
            std::cout<<"\t phiBz: "<<phiBz<<std::endl;
            std::cout<<"\t E0x: "<<E0x<<std::endl;
            std::cout<<"\t E0z: "<<E0z<<std::endl;
            std::cout<<"\t B0x: "<<B0x<<std::endl;
            std::cout<<"\t B0z: "<<B0z<<std::endl;
        }
    };

#endif
