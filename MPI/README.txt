Inside this folder you can find 5 files, mpi_fft.c (code), mpi_fft (compile file), fft_mpi.sbatch (configure file slurm) and ouput_mpi_fft.out (results).


this code has been tested in local and guane for N = 20

Using opnempi version 3.1.4


Compilation
mpicc mpi_fft.c -o mpi_fft -lm -std=gnu99

Execution

sbatch fft-mpi.sbatch 

