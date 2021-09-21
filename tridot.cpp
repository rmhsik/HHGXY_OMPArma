#include <cmath>

#include "tridot.h"

void tridot(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N){
        (*out)[0] = (*a)[0]*(*in)[0] + (*b)[0]*(*in)[1] ;
        (*out)[N-1] = (*c)[N-1]*(*in)[N-2] + (*b)[N-1]*(*in)[N-1];

        for(int i = 1; i<N-1; i++){
            (*out)[i] = (*c)[i]*(*in)[i-1] + (*b)[i]*(*in)[i] + (*a)[i]*(*in)[i+1];
        }
}

void tridotBatched(std::complex<double> **a, std::complex<double> **b, std::complex<double> **c, std::complex<double> **in, std::complex<double> **out, const int N,const int m, const int id){
	const int first = id*N;
	const int last = N*id + N;
        (*out)[first] = (*b)[first]*(*in)[first] + (*c)[first]*(*in)[first+1] ;
        (*out)[last-1] = (*a)[last-1]*(*in)[last-2] + (*b)[last-1]*(*in)[last-1];

        for(int i = first+1; i<last-1; i++){
            (*out)[i] = (*a)[i]*(*in)[i-1] + (*b)[i]*(*in)[i] + (*c)[i]*(*in)[i+1];
        }
}
void tridotBatchedZ(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id){
	const int first = id*N;
	const int last = N*id + N;
        PsiOut(id,0) = b(0,id)*Psi(id,0) + c(0,id)*Psi(id,1);
        PsiOut(id,N-1) = a(N-1,id)*Psi(id,N-2) + b(N-1,id)*Psi(id,N-1);
	
	
        for(int i = 1; i<N-1; i++){
            PsiOut(id,i) = a(i,id)*Psi(id,i-1) + b(i,id)*Psi(id,i) + c(i,id)*Psi(id,i+1);
        }
}
void tridotBatchedR(arma::cx_mat &a, arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &Psi, arma::cx_mat &PsiOut, const int N, const int m, const int id){
	const int first = id*N;
	const int last = N*id + N;
        PsiOut(0,id) = b(0,id)*Psi(0,id) + c(0,id)*Psi(1,id);
        PsiOut(N-1,id) = a(N-1,id)*Psi(N-2,id) + b(N-1,id)*Psi(N-1,id);
	
	if(id<m){	
        for(int i = 1; i<N-1; i++){
            PsiOut(i,id) = a(i,id)*Psi(i-1,id) + b(i,id)*Psi(i,id) + c(i,id)*Psi(i+1,id);
        }}
	else{
	    std::cout<<"ERROR\n";
	}
}

void tridot(arma::cx_mat &H, arma::cx_colvec &Psi,
            arma::cx_colvec &Psiout, const int N){

        Psiout(0) = H.col(1)(0)*Psi(0) + H.col(0)(0)*Psi(1) ;
        Psiout(N-1) = H.col(2)(N-1)*Psi(N-2) + H.col(1)(N-1)*Psi(N-1);

        for(int i = 1; i<N-1; i++){
            Psiout(i) = H.col(2)(i)*Psi(i-1) + H.col(1)(i)*Psi(i) + H.col(0)(i)*Psi(i+1);
        }
}

void tdmaSolver(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N){

    std::complex<double> wc;
    arma::cx_colvec ac = H.col(2);
    arma::cx_colvec bc = H.col(1);
    arma::cx_colvec cc = H.col(0);
    arma::cx_colvec dc = Psi;

    for(int i = 1; i<=N-1;i++){
        wc = ac(i)/bc(i-1);
        bc(i) = bc(i) - wc*cc(i-1);
        dc(i) = dc(i) - wc*dc(i-1);
    }

    Psiout(N-1) = dc(N-1)/bc(N-1);
    
    for(int i = N-2;i>=0;i--){
        Psiout(i) = (dc(i)-cc(i)*Psiout(i+1))/bc(i);
    }
}

/*
void tridot(arma::cx_mat &H, arma::cx_colvec &Psi, 
            arma::cx_colvec &Psiout, const int N){

        Psiout(0) = H.col(1)(0)*Psi(0) + H.col(0)(0)*Psi(1) ;
        Psiout(N-1) = H.col(2)(N-1)*Psi(N-2) + H.col(1)(N-1)*Psi(N-1);

        for(int i = 1; i<N-1; i++){
            Psiout(i) = H.col(2)(i)*Psi(i-1) + H.col(1)(i)*Psi(i) + H.col(0)(i)*Psi(i+1);
        }
}*/
