#include <cmath>
#include "../include/fields.h"

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

double EField(double t){
    double E0=0.067;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    return E0*envelope_sin2(tmax,t,w)*sin(w*t);
}

double BField(double t){
    double B0=0.00;
    double w = 0.057;
    double tmax = 4*2*M_PI/w;
    double phi = 0.12*M_PI;
    return B0*envelope_sin2(tmax,t,w)*sin(w*t+phi);
}


