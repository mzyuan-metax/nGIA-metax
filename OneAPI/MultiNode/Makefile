MPI = -I"/opt/intel/oneapi/mpi/2021.4.0/include" -L"/opt/intel/oneapi/mpi/2021.4.0/lib" -lmpi
all:
	dpcpp $(MPI) src/cluster.cpp src/func.cpp -o cluster
run:
	mpirun -n 2 ./cluster -i .. -o .. -w 8 -s 0.95
