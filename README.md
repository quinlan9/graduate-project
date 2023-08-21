# graduate-project


scp qinf@rsync.tchpc.tcd.ie:~/NMF/NMF.c  qinf@chuck.tchpc.tcd.ie:~/NMF  
scp qinf@chuck.tchpc.tcd.ie:~/NMF/NMF.c  qinf@rsync.tchpc.tcd.ie:~/NMF  

serial:    
	gcc NMFserial.c -std=c99 -o NMFserial  
	./NMFserial  
MPI:  
	mpicc -g -o NMFMPI NMFMPI.c -lm -std=c99  
	mpirun -np 1 ./NMFMPI  
OpenMP:  
	mpicc -g -o NMFhybrid NMFhybrid.c -lm -fopenmp -std=c99  
mpirun -np 4 -x OMP_NUM_THREADS=3 ./NMFhybrid   
