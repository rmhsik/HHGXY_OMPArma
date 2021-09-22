#ifndef FIELDS_H
#define FIELDS_H
typedef struct parameters parameters;
double envelope_sin2(double tmax, double t,double w);
double envelope_trap(double tmax, double t, double w);
double EField(double t,parameters p);
double BField(double t, parameters p);
#endif
