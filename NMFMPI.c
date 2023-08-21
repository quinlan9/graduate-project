#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include <string.h>
#include <math.h>

//void readMatrixFromFile(const char* file_name, int m, int n, double* V);
void computeTranspose(const double* V, int m, int n, double* V_T);
void init_matrix(double *W,int m,int r);
void matrixMultiplication(const double* T, const double* W, int c, int m, int r, double* M);
void update_H(double* H,double* W_T,double* Vblock,double * W_T_W, int r, int c, int m);
void update_W(double* Wblock, const double* Vblock, const double* H_T, const double* HHT, int c, int r, int n);
double calculate_diff(double* V, double* WH,int m,int r,int n);
void printMatrix(double* matrix, int m, int n);
//void writeMatrixToFile(const char *filename, double *matrix, int rows, int cols);

int main(int argc, char** argv) {

	int m=4200;
	int n=4200;
	int r=72;
	int maxit=10;

    double* V = (double*)malloc(m * n * sizeof(double));
   	double* W = (double*)malloc(m * r * sizeof(double));
   	double* H = (double*)malloc(r * n * sizeof(double));
   	double* WH = (double*)malloc(m * n * sizeof(double));
    double* W_T = (double*)malloc(r * m * sizeof(double));
    double* H_T = (double*)malloc(n * r * sizeof(double));
    double* W_T_W = (double*)malloc(r * r * sizeof(double));
    double* H_H_T = (double*)malloc(r * r * sizeof(double));
    int myid, nprocs;
    int it;
	double diff;
	double t1,t2;
    double tol = 0.05;
	/*initialize MPI*/
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /*initialize original matrix V*/
    if (V == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

	if(myid==0){
		init_matrix(V,m,n);
	}

    /*calculate number of block of each matrix*/
    //assume everything is divisable by process
    int V_colb_num = n / nprocs;
    int H_colb_num = V_colb_num;

    int V_rowb_num = m / nprocs;
    int W_rowb_num = V_rowb_num;

    //used for divide matrix
    double* Vt = (double*)malloc(m * n * sizeof(double));  //V transform
    double*blockVc=(double*)malloc(m*V_colb_num*sizeof(double));    
    double* Ht = (double*)malloc(r * n * sizeof(double));  //H transform
	double*blockHc=(double*)malloc(r*H_colb_num*sizeof(double)); 


    double* blockVcT = (double*)malloc(m * V_colb_num * sizeof(double)); 
    double* blockHcT = (double*)malloc(r * H_colb_num * sizeof(double));

    double* blockVr = (double*)malloc(n * V_rowb_num * sizeof(double));
    double* blockWr = (double*)malloc(r * W_rowb_num * sizeof(double));

    //Row decomposition method to find the transposed product
    double* blockW_Tr = (double*)malloc(m * (r / nprocs) * sizeof(double));
    double* blockHr = (double*)malloc(n * (r / nprocs) * sizeof(double));

    double* blockW_Tr_W = (double*)malloc(r * (r / nprocs) * sizeof(double));
    double* blockHr_H_T = (double*)malloc(r * (r / nprocs) * sizeof(double));

    /*only the root process initilaize the full matrix*/
    	if (myid == 0) {
        //generate the initial access W,H
        	init_matrix(W,m,r);
        	init_matrix(H,r,n);    

    	}
 
   //send matrix block V_col and H_col
    computeTranspose(V,m,n,Vt);
	computeTranspose(H,r,n,Ht);

    MPI_Scatter(Vt, m*V_colb_num, MPI_DOUBLE, blockVcT, m * V_colb_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(Ht, r * H_colb_num, MPI_DOUBLE, blockHcT, r * H_colb_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    computeTranspose(blockVcT,V_colb_num,m,blockVc);//get blockVc
    computeTranspose(blockHcT,H_colb_num,r,blockHc);

    
    //send matrix block V_row and W_row (verified)
    MPI_Scatter(V, n * V_rowb_num, MPI_DOUBLE, blockVr, n * V_rowb_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(W, r * W_rowb_num, MPI_DOUBLE, blockWr, r * W_rowb_num, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double global_diff=1000;
	t1 = MPI_Wtime(); //get current time
	for (it = 0; it < maxit; it++) {

        //broadcast W to all process
        MPI_Bcast(W, m * r, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        /*stage1: calculate H*/
        if (myid == 0) {
            //compute W_T
            computeTranspose(W, m, r, W_T);
        }

	    MPI_Bcast(W_T, m * r, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Scatter matrix W_T to all processes
        MPI_Scatter(W_T, m*(r / nprocs), MPI_DOUBLE, blockW_Tr,m*(r / nprocs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform matrix multiplication
        matrixMultiplication(blockW_Tr, W, r / nprocs, m, r, blockW_Tr_W);

        // Gather partial results from all processes
        MPI_Gather(blockW_Tr_W, r * (r / nprocs), MPI_DOUBLE, W_T_W, r * (r / nprocs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //broadcast W_T_W to all processes
        MPI_Bcast(W_T_W, r * r, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //calculate h
        update_H(blockHc, W_T, blockVc, W_T_W, r, n / nprocs, m);

        computeTranspose(blockHc,r,H_colb_num,blockHcT);

        //gather the new entire H
        MPI_Gather(blockHcT, r * n/nprocs, MPI_DOUBLE, H_T, r * n/nprocs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (myid == 0) {
	    computeTranspose(H_T,n,r,H);	
        }
//--------------------------------------------------------------------------------
        
        /*Broadcast H_T*/
        MPI_Bcast(H_T, r * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

 
        MPI_Scatter(H, n*(r / nprocs), MPI_DOUBLE, blockHr, n*(r / nprocs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Perform matrix multiplication
        matrixMultiplication(blockHr, H_T, r / nprocs, n, r, blockHr_H_T);

        // Gather partial results from all processes
        MPI_Gather(blockHr_H_T, r * (r / nprocs), MPI_DOUBLE, H_H_T, r * (r / nprocs), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //broadcast H_H_T to all processes
        MPI_Bcast(H_H_T, r * r, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
        //calculate W
        update_W(blockWr, blockVr, H_T, H_H_T, m / nprocs, r, n);

        //gather the new entire W
        MPI_Gather(blockWr, r* m/nprocs, MPI_DOUBLE, W, r* m/nprocs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //compute WH
        matrixMultiplication(W, H, m, r, n, WH);

        //calculate the difference

        diff=calculate_diff(V, WH ,m, r, n);

	    MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	

	    if(myid==0){  
        	printf("global_diff:%lf\n",diff);
	    }
//        printf("global_diff:%lf\n",global_diff);

        /*Break the loop if the difference is below the tolerance*/
        if (global_diff < tol) {
            break;
        }
}
	t2 = MPI_Wtime();
    if(myid==0){
    printf("it:%d\n",it);
    printf("diff:%lf\n",diff);
    printf("global_diff:%lf\n",global_diff);
	printf("Run took %lf s\n", t2 - t1);

    }
    free(V);
    free(W);
    free(H);
    free(W_T);
    free(H_T);
    free(W_T_W);
    free(H_H_T);
    free(blockHcT);
    free(blockHr);
    free(blockVcT);
    free(blockHr_H_T);
    free(blockW_Tr_W);
    free(blockW_Tr);
    free(blockVr);
    free(blockWr);
    free(WH);
    free(Vt);
    free(blockVc);
    free(Ht);
    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

/*
void readMatrixFromFile(const char* file_name, int m, int n, double* V) {
    // Open the file for reading
    FILE* file = fopen(file_name, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return;
    }
    // Read the values from the file into the matrix
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fscanf(file, "%lf", &V[i * n + j]) != 1) {
                printf("Failed to read value at row %d, column %d.\n", i, j);
                fclose(file);
                return;
            }
        }
    }
    // Close the file
    fclose(file);
}
*/



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



void update_H(double* H,double* W_T,double* Vblock,double * W_T_W, int r, int c, int m) {

    double* W_T_V = (double*)malloc(r * c * sizeof(double));
    double* W_T_W_H = (double*)malloc(r * c * sizeof(double));

    //compute W_T * V
	matrixMultiplication(W_T,Vblock,r,m,c,W_T_V);

    //compute W_T_W * H
	matrixMultiplication(W_T_W,H,r,r,c,W_T_W_H);

    //compute H
    // Compute H * (W_T * V) / (W_T * W * H)
    for (int i = 0; i < r * c; i++) {
        H[i] *= (W_T_V[i] / W_T_W_H[i]);
    }

    free(W_T_V);
    free(W_T_W_H);
}


void update_W(double* Wblock, const double* Vblock, const double* H_T, const double* HHT, int c, int r, int n) {
    // Allocate memory for intermediate matrices
    double* VH_T = (double*)malloc(c * r * sizeof(double));
    double* WHHT = (double*)malloc(c * r * sizeof(double));


    matrixMultiplication(Vblock,H_T,c,n,r,VH_T);
    matrixMultiplication(Wblock,HHT,c,r,r,WHHT);
    // Compute V * H_T

    // Update W based on the equation: W = W * (V * H_T) / (W * HHT)
    for (int i = 0; i < c * r; i++) {
        Wblock[i] = Wblock[i] * (VH_T[i] / WHHT[i]);
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


/*void writeMatrixToFile(const char *filename, double *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("failed to open %s\n", filename);
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%f ", matrix[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}
*/
