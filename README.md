# graduate-project

serial:    
	gcc NMFserial.c -std=c99 -o NMFserial  
	./NMFserial  
MPI:  
	mpicc -g -o NMFMPI NMFMPI.c -lm -std=c99  
	mpirun -np 1 ./NMFMPI  
hybrid:  
	mpicc -g -o NMFhybrid NMFhybrid.c -lm -fopenmp -std=c99  
mpirun -np 4 -x OMP_NUM_THREADS=3 ./NMFhybrid   
