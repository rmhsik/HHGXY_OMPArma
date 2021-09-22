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

double EField(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0E;
    if (p.env==0){
        return p.E0*envelope_sin2(tmax,t,p.w0E)*sin(p.w0E*t+p.phiE);
    }
    else if (p.env==1){
        return p.E0*envelope_trap(tmax,t,p.w0E)*sin(p.w0E*t+p.phiE);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0; 
    }
}

double BField(double t,parameters p){
    double tmax = p.fieldPeriods*2*M_PI/p.w0B;
    if (p.env==0){
        return p.B0*envelope_sin2(tmax,t,p.w0B)*sin(p.w0B*t+p.phiB);
    }
    else if (p.env==1){
        return p.B0*envelope_trap(tmax,t,p.w0B)*sin(p.w0B*t+p.phiB);
    }
    else{
        std::cout<<"[ERROR] Bad envelope definition in param.cpp\n";
        return 0.0;
    }
}


