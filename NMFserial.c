#include<stdio.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>
#include<time.h>


void computeTranspose(const double* V, int m, int n, double* V_T);
void init_matrix(double *W,int m,int r);
void matrixMultiplication(const double* T, const double* W, int c, int m, int r, double* M);
void update_H(double* H,double* W_T,double* V,double * W_T_W, int r, int n, int m);
void update_W(double* W, const double* V, const double* H_T, const double* HHT, int m, int r, int n);
double calculate_diff(double* V, double* WH,int m,int r,int n);
void printMatrix(double* matrix, int m, int n);

int main(int argc, char** argv) {

	int m=4200;
	int n=4200;
	int r=72;
	int maxit=10;
    int it;
    double diff=1000.0;
   
clock_t start, end;
    double tol = 0.05;
    
    double* V = (double*)malloc(m * n * sizeof(double));//
    double* W = (double*)malloc(m * r * sizeof(double));//
    double* H = (double*)malloc(r * n * sizeof(double));//

    double* WH = (double*)malloc(m * n * sizeof(double));//
    double* W_T = (double*)malloc(r * m * sizeof(double));//
    double* H_T = (double*)malloc(n * r * sizeof(double));
    double* W_T_W = (double*)malloc(r * r * sizeof(double));
    double* H_H_T = (double*)malloc(r * r * sizeof(double));

    init_matrix(V, m, n);
    init_matrix(W, m, r);
    init_matrix(H, r, n);

    start = clock();
    for (it = 0; it < maxit; it++) {

        computeTranspose(W, m, r, W_T);
        matrixMultiplication(W_T, W, r, m, r, W_T_W);

        update_H(H, W_T, V, W_T_W, r, n, m);

        computeTranspose(H, r, n, H_T);
        matrixMultiplication(H, H_T, r, n, r, H_H_T);
        update_W(W, V, H_T, H_H_T, m, r, n);

        matrixMultiplication(W, H, m, r, n, WH);
        diff = calculate_diff(V, WH, m, r, n);
        printf("diff:%lf\n",diff);
        /*Break the loop if the difference is below the tolerance*/
        if (diff < tol) {
            break;
        }
    }

    end = clock();
    
    printf("it:%d\n",it);
    printf("diff:%lf\n",diff);
    double cpu_time_used;
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Run took %lf s\n", cpu_time_used);

    
    free(V);
    free(W);
    free(H);
    free(W_T);
    free(H_T);
    free(W_T_W);
    free(H_H_T);
    free(WH);
    
    return 0;
}


/*compute the transform of the matrix*/
 void computeTranspose(const double* V, int m, int n, double* V_T) {
        // Compute the transpose of V
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            V_T[j * m + i] = V[i * n + j];
            }
        }
    }


void init_matrix(double *W,int m,int r) {
    for (int i=0; i < m; i++) {
        for (int j = 0; j < r; j++) {
            double randomValue = ((double)rand() / RAND_MAX);
            W[i * r + j] = randomValue;
        }
    }
}


/*function to calculate the product of two matrices */
void matrixMultiplication(const double* T, const double* W, int c, int m, int r, double* M) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
            M[i * r + j] = 0.0;
            for (int k = 0; k < m; k++) {
                M[i * r + j] += T[i * m + k] * W[k * r + j];
            }
        }
    }
}



void update_H(double* H,double* W_T,double* V,double * W_T_W, int r, int n, int m) {

    double* W_T_V = (double*)malloc(r * n * sizeof(double));
    double* W_T_W_H = (double*)malloc(r * n * sizeof(double));

    //compute W_T * V
	matrixMultiplication(W_T,V,r,m,n,W_T_V);

    //compute W_T_W * H
	matrixMultiplication(W_T_W,H,r,r,n,W_T_W_H);

    //compute H
    // Compute H * (W_T * V) / (W_T * W * H)
    for (int i = 0; i < r * n; i++) {
        H[i] *= (W_T_V[i] / W_T_W_H[i]);
    }

    free(W_T_V);
    free(W_T_W_H);
}


void update_W(double* W, const double* V, const double* H_T, const double* HHT, int m , int r, int n) {
    // Allocate memory for intermediate matrices
    double* VH_T = (double*)malloc(m * r * sizeof(double));
    double* WHHT = (double*)malloc(m * r * sizeof(double));


    matrixMultiplication(V,H_T,m,n,r,VH_T);
    matrixMultiplication(W,HHT,m,r,r,WHHT);

    // Update W based on the equation: W = W * (V * H_T) / (W * HHT)
    for (int i = 0; i < m * r; i++) {
        W[i] = W[i] * (VH_T[i] / WHHT[i]);
    }

    // Free the allocated memory
    free(VH_T);
    free(WHHT);
}

double calculate_diff(double* V, double* WH,int m,int r,int n) {  
	double diff=0.0;
	double Vsum=0.0;
    for (int i = 0; i < m * n; i++) {
 		diff += (V[i] - WH[i])*(V[i] - WH[i]);
	}
	
	for(int i=0;i<m*n;i++){
		Vsum += V[i];		
	}
	return fabs(diff/Vsum);
}

void printMatrix(double* matrix, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            // Access the element at position (i, j) using pointer arithmetic
            double element = *(matrix + i * n + j);
            printf("%lf ", element);
        }
        printf("\n");
    }
}



