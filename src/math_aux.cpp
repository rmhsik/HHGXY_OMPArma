#include <cmath>
#include "math_aux.h"
#include "param.h"
void Gaussian(arma::cx_mat &Psi, arma::vec &x, arma::vec &z, const double x0,const double z0, const double a ){
    for(int i=0;i<Psi.n_rows;i++){
        for(int j=0;j<Psi.n_cols;j++){
            Psi(i,j) = exp(-pow(x(i),2)/a-pow(z(j)-z0,2)/a);
        }
    }
}

void derivativeZ(arma::dmat &U, arma::dmat z, arma::dmat &DU){
    int Nz = z.n_elem;
    double dz = (z(z.n_elem-1)-z(0))/(double)Nz;
    
    for(int i=0;i<U.n_rows;i++){
        DU(i,0) = (U(i,1)-U(i,0))/dz;
    }
    for(int i=0;i<U.n_rows;i++){
        DU(i,U.n_cols-1) = (U(i,U.n_cols-1)-U(i,U.n_cols-2))/dz;
    }

    for(int i=0;i<U.n_rows;i++){
        for(int j=1; j<U.n_cols-1;j++){
            DU(i,j) = (U(i,j+1)-U(i,j-1))/(2.0*dz);
        }
    }
}
void derivativeX(arma::dmat &U, arma::dmat x, arma::dmat &DU){
    int Nx = x.n_elem;
    double dx = (x(x.n_elem-1)-x(0))/(double)Nx;
    
    for(int i=0;i<U.n_cols;i++){
        DU(0,i) = (U(1,i)-U(0,i))/dx;
    }
    for(int i=0;i<U.n_cols;i++){
        DU(U.n_rows-1,i) = (U(U.n_rows-1,i)-U(U.n_rows-2,i))/dx;
    }

    for(int i=1;i<U.n_rows-1;i++){
        for(int j=0; j<U.n_cols;j++){
            DU(i,j) = (U(i+1,j)-U(i-1,j))/(2.0*dx);
        }
    }
}



void tdmaSolver(double **a ,double **b, double **c, double **d, 
            double **out, const int N){

    double wc;
   for(int i = 1; i<=N-1;i++){
        wc = (*a)[i]/(*b)[i-1];
        (*b)[i] = (*b)[i] - wc*(*c)[i-1];
        (*d)[i] = (*d)[i] - wc*(*d)[i-1];
    }

    (*out)[N-1] = (*d)[N-1]/(*b)[N-1];
    
    for(int i = N-2;i>=0;i--){
        (*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
    }
}


void tdmaSolverBatch(double **a ,double **b, double **c, double **d, 
            double **out, const int N,const int m, const int id){
	if (id<m){
	   double wc;
	   int first = id*N;
       int last = N*id + N;
	   std::cout<<"first: "<<first<<std::endl;
	   std::cout<<"last: "<<last<<std::endl;
	   for(int i = first+1; i<=last-1;i++){
			wc = (*a)[i]/(*b)[i-1];
			(*b)[i] = (*b)[i] - wc*(*c)[i-1];
			(*d)[i] = (*d)[i] - wc*(*d)[i-1];
		}

		(*out)[last-1] = (*d)[last-1]/(*b)[last-1];
		
		for(int i = last-2;i>=first;i--){
			(*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
		}
	}
}


void tdmaSolverBatchC(std::complex<double> **a ,std::complex<double> **b, std::complex<double> **c, std::complex <double> **d, std::complex<double> **out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       int last = N*id + N;
	   for(int i = first+1; i<=last-1;i++){
			wc = (*a)[i]/(*b)[i-1];
			(*b)[i] = (*b)[i] - wc*(*c)[i-1];
			(*d)[i] = (*d)[i] - wc*(*d)[i-1];
		}

		(*out)[last-1] = (*d)[last-1]/(*b)[last-1];
		
		for(int i = last-2;i>=first;i--){
			(*out)[i] = ((*d)[i]-(*c)[i]*(*out)[i+1])/(*b)[i];
		}
	}
}

void tdmaSolverBatchZ(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       	   int last = N*id + N;
	   for(int i = 1; i<=N-1;i++){
			wc = a(i,id)/b(i-1,id);
			b(i,id) = b(i,id) - wc*c(i-1,id);
			d(id,i) = d(id,i) - wc*d(id,i-1);
		}

		out(id,N-1) = d(id,N-1)/b(N-1,id);
		
		for(int i = N-2;i>=first;i--){
			out(id,i) = (d(id,i)-c(i,id)*out(id,i+1))/b(i,id);
		}
	}
}

void tdmaSolverBatchR(arma::cx_mat &a , arma::cx_mat &b, arma::cx_mat &c, arma::cx_mat &d, arma::cx_mat &out, const int N,const int m, const int id){
	if (id<m){
	   std::complex<double> wc;
	   int first = id*N;
       	   int last = N*id + N;
	   for(int i = 1; i<=N-1;i++){
			wc = a(i,id)/b(i-1,id);
			b(i,id) = b(i,id) - wc*c(i-1,id);
			d(i,id) = d(i,id) - wc*d(i-1,id);
		}

		out(N-1,id) = d(N-1,id)/b(N-1,id);
		
		for(int i = N-2;i>=first;i--){
			out(i,id) = (d(i,id)-c(i,id)*out(i+1,id))/b(i,id);
		}
	}
}

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

double intSimpson(double (*func)(double,parameters), double from, double to, int n, parameters p){
    // https://rosettacode.org/wiki/Numerical_integration#C
    
    double h = (to - from) / n;
    double sum1 = 0.0;
    double sum2 = 0.0;
    int i;
 
    double x;
 
    for(i = 0;i < n;i++)
        sum1 += func(from + h * i + h / 2.0,p);
 
    for(i = 1;i < n;i++)
        sum2 += func(from + h * i,p);
 
    return h / 6.0 * (func(from,p) + func(to,p) + 4.0 * sum1 + 2.0 * sum2);
}
