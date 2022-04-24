MPI = -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
all:
	nvcc $(MPI) src/cluster.cpp src/func.cu -o cluster
run:
	mpirun -n 2 ./cluster -i .. -o .. -w 3 -s 0.95
