#ifndef PARAM_H
#define PARAM_H
#include <cmath>
#include <iostream>
    const double c = 137.04;

    struct parameters{
        double rmin     = 0.0;
        double rmax     = 150.0;
        int Nr          = 1500;
        double dr       = (rmax-rmin)/Nr;
        double zmin     = -120.0;
        double zmax     =  120.0;
        int Nz          = 2400; 
        double dz       = (zmax-zmin)/Nz;
        
        double w0E      = 0.057;
        double w0B      = 0.051;
        double phiE     = 0.0;
        double phiB     = -0.25*M_PI;
        double E0       = 0.067;
        double B0       = 0.12;
        int env         = 1; //0 -> sin2; 1 -> trap 
        
        double t0       = 0.0;
        double dt       = 0.02;
        double simPeriods   = 6.0;
        double fieldPeriods = 4.0;
        int Nsteps      = 50;
	    double tmax;	
        int Nt;
        double dt_ITP   = 0.001;
        int Nt_ITP      = 300;
        
        parameters(){
            if(w0B<=w0E){
                tmax = simPeriods*2.0*M_PI/w0B;
            }
            else{ 
                tmax = simPeriods*2.0*M_PI/w0E;
            }
            Nt = (tmax-t0)/dt;
        }
        void printParameters(){
            std::cout<<"Parameters:\n";
            std::cout<<"-----------\n";
            std::cout<<"\t rmin: "<<rmin<<std::endl;
            std::cout<<"\t rmax: "<<rmax<<std::endl;
            std::cout<<"\t Nr: "<<Nr<<std::endl;
            std::cout<<"\t dr: "<<dr<<std::endl;
            std::cout<<"\t zmin: "<<zmin<<std::endl;
            std::cout<<"\t zmax: "<<zmax<<std::endl;
            std::cout<<"\t Nz: "<<Nz<<std::endl;
            std::cout<<"\t dz: "<<dz<<std::endl;
            std::cout<<"\t w0E: "<<w0E<<std::endl;
            std::cout<<"\t w0B: "<<w0B<<std::endl;
            std::cout<<"\t t0: "<<t0<<std::endl;
            std::cout<<"\t tmax: "<<tmax<<std::endl;
            std::cout<<"\t dt: "<<dt<<std::endl;           
            std::cout<<"\t Nt: "<<Nt<<std::endl;
            std::cout<<"\t phiE: "<<phiE<<std::endl;
            std::cout<<"\t phiB: "<<phiB<<std::endl;
            std::cout<<"\t E0: "<<E0<<std::endl;
            std::cout<<"\t B0: "<<B0<<std::endl;
        }
    };

#endif
