Inside this folder you can find 4 files, omp_fft.c (code), omp_fft (compile file), fft_omp.sbatch (configure file slurm) and outpu_fft_omp.txt (results).


this code has been tested in local and guane for N = 28

Using gcc version 8.3.0

Local: 8 cores

Compilation
gcc -fopenmp omp_fft.c -o omp_fft -lm

Execution
./omp_fft


GUANE: 16 cores

Compilation
gcc -fopenmp omp_fft.c -o omp_fft -lm

Execution

sbatch omp_fft.sbatch

