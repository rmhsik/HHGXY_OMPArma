#ifndef FIELDS_H
#define FIELDS_H
typedef struct parameters parameters;
double envelope_sin2(double tmax, double t,double w);
double envelope_trap(double tmax, double t, double w);
double EFieldX(double t,parameters p);
double EFieldZ(double t,parameters p);
double BFieldX(double t, parameters p);
double BFieldZ(double t, parameters p);
#endif
