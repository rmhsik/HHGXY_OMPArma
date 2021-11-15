#include <cmath>
#include <iostream>
#include "fields.h"
#include "param.h"

double envelope_sin2(double tmax, double t,double w){
    if (t<tmax){
        return pow(sin(M_PI*t/tmax),2);
    }
    else {
        return 0.0;
    }
}

double envelope_trap(double tmax, double t, double w){
    double T = 2*M_PI/w;
    if (t<tmax+2.0*T){
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

double EFieldZ(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0Ez;
    if (p.env==0){
        return p.E0z*envelope_sin2(tmax,t,p.w0Ez)*sin(p.w0Ez*t+p.phiEz);
    }
    else if (p.env==1){
        return p.E0z*envelope_trap(tmax,t,p.w0Ez)*sin(p.w0Ez*t+p.phiEz);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0; 
    }
}

double EFieldX(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0Ex;
    if (p.env==0){
        return p.E0x*envelope_sin2(tmax,t,p.w0Ex)*sin(p.w0Ex*t+p.phiEx);
    }
    else if (p.env==1){
        return p.E0x*envelope_trap(tmax,t,p.w0Ex)*sin(p.w0Ex*t+p.phiEx);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0; 
    }
}

double BFieldX(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0Bx;
    if (p.env==0){
        return p.B0x*envelope_sin2(tmax,t,p.w0Bx)*sin(p.w0Bx*t+p.phiBx);
    }
    else if (p.env==1){
        return p.B0x*envelope_trap(tmax,t,p.w0Bx)*sin(p.w0Bx*t+p.phiBx);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0;
    }
}

double BFieldZ(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0Bz;
    if (p.env==0){
        return p.B0z*envelope_sin2(tmax,t,p.w0Bz)*sin(p.w0Bz*t+p.phiBz);
    }
    else if (p.env==1){
        return p.B0z*envelope_trap(tmax,t,p.w0Bz)*sin(p.w0Bz*t+p.phiBz);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0;
    }
}

