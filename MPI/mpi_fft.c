#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <float.h>

int main(int argc, char **argv);
void printResponse(int n, int nits, int error, double ctime, int mflops);
void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time(void);
double ggl(double *ds);
void step(int n, int mj, double a[], double b[], double c[], double d[],
    double w[], double sgn);
void timestamp();

//declare TAG for use in methods mpi Send and Recv
#define MASTER 0        /* task ID of master task */
#define TAG_N 1
#define TAG_NITS 2
#define TAG_NITS2 3
#define TAG_ERROR 4
#define TAG_TIME 5
#define TAG_MFLOPS 6

/******************************************************************************/

int main(int argc, char **argv)

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for FFT_SERIAL.

  Discussion:

    The "complex" vector A is actually stored as a double vector B.

    The "complex" vector entry A[I] is stored as:

      B[I*2+0], the real part,
      B[I*2+1], the imaginary part.

  Modified:

    23 March 2009

  Author:

    Original C version by Wesley Petersen.
    This C version by John Burkardt.

  Reference:

    Wesley Petersen, Peter Arbenz,
    Introduction to Parallel Computing - A practical guide with examples in C,
    Oxford University Press,
    ISBN: 0-19-851576-6,
    LC: QA76.58.P47.
*/
{
    double ctime;
    double ctime1;
    double ctime2;
    double error;
    int first;
    double flops;
    double fnm1;
    int i;
    int icase;
    int it;
    int ln2;
    double mflops;
    int n;
    int nits;
    static double seed;
    double sgn;
    double *w;
    double *x;
    double *y;
    double *z;
    double z0;
    double z1;
    int length = 100;

    //variables for MPI use
    int ntasks;  /* task ID */
    int rank; /* number of tasks */
    int dest, source;
    double wtime;
    double starttime, endtime;

    //Get the Status
    MPI_Status Stat;

    MPI_Init(&argc, &argv);

    //get the total number of MPI process
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
    //get the total number of calling MPI process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //get an elapsed wall clock time in seconds 
    starttime = MPI_Wtime();


    nits = 10000;
    seed = 331.0;
    n = 1;
    int total = 20;
    int inicio = 1;

    
    //just master does this.
    if (rank==MASTER)//master process
    {
        // timestamp();
        printf("\n");
        printf("FFT_MPI\n");
        printf("  MPI version\n");
        printf("\n");
        printf("  Demonstrate an implementation of the Fast Fourier Transform\n");
        printf("  of a complex data vector.\n");
        /*
    Prepare for tests.
    */
        printf("\n");
        printf("  Accuracy check:\n");
        printf("\n");
        printf("    FFT ( FFT ( X(1:N) ) ) == N * X(1:N)\n");
        printf("\n");
        printf("             N      NITS    Error         Time          Time/Call     MFLOPS\n");
        printf("\n");
        dest = 1;
        source = 1;    
        
        //for each iter calling a print data to return each process
        for (int i = 1; i < total; i++)
        {

          //recive results for workers process
          MPI_Recv(&n, 1, MPI_INT, source, TAG_N, MPI_COMM_WORLD, &Stat);
          MPI_Recv(&nits, 1, MPI_INT, source, TAG_NITS, MPI_COMM_WORLD, &Stat);
          MPI_Recv(&error, 1, MPI_DOUBLE, source, TAG_ERROR, MPI_COMM_WORLD, &Stat);
          // MPI_Recv(&ctime, 1, MPI_INT, source, TAG_TIME, MPI_COMM_WORLD, &Stat);
          // MPI_Recv(&mflops, 1, MPI_DOUBLE, source, TAG_MFLOPS, MPI_COMM_WORLD, &Stat);
          printf("  %12d  %8d  %12e", n, nits, error);
          printf("  %12e  %12e  %12f\n", ctime, ctime / (double)(2 * nits), mflops);   
        }
        
              
    }
    else{ // workes process and nodes
     

        /*
    LN2 is the log base 2 of N.  Each increase of LN2 doubles N.

    */  

      //load balancing with this for, each procees just does it parcial for and then return result to master process 
      for (int r = 1; r <= rank; r++){
        if (rank==r)
        {
          for (ln2 = inicio; ln2 < total/rank*r; ln2++)
        {
            n = 2 * n;
            /*
      Allocate storage for the complex arrays W, X, Y, Z.

      We handle the complex arithmetic,
      and store a complex number as a pair of doubles, a complex vector as a doubly
      dimensioned array whose second dimension is 2.
    */
            w = (double *)malloc(n * sizeof(double));
            x = (double *)malloc(2 * n * sizeof(double));
            y = (double *)malloc(2 * n * sizeof(double));
            z = (double *)malloc(2 * n * sizeof(double));

            first = 1;

            for (icase = 0; icase < 2; icase++)
            {
                if (first)
                {
                    for (i = 0; i < 2 * n; i = i + 2)
                    {
                        z0 = ggl(&seed);
                        z1 = ggl(&seed);
                        x[i] = z0;
                        z[i] = z0;
                        x[i + 1] = z1;
                        z[i + 1] = z1;
                    }
                }
                else
                {
                    for (i = 0; i < 2 * n; i = i + 2)
                    {
                        z0 = 0.0; /* real part of array */
                        z1 = 0.0; /* imaginary part of array */
                        x[i] = z0;
                        z[i] = z0; /* copy of initial real data */
                        x[i + 1] = z1;
                        z[i + 1] = z1; /* copy of initial imag. data */
                    }
                }
                /*
        Initialize the sine and cosine tables.
      */
                cffti(n, w);
                /*
        Transform forward, back
      */
                if (first)
                {
                    dest = 0;
                    source = 0;
                    sgn = +1.0;
                    cfft2(n, x, y, w, sgn);
                    sgn = -1.0;
                    cfft2(n, y, x, w, sgn);
                    /*
          Results should be same as the initial data multiplied by N.
        */
                    fnm1 = 1.0 / (double)n;
                    error = 0.0;
                    for (i = 0; i < 2 * n; i = i + 2)
                    {
                        error = error + pow(z[i] - fnm1 * x[i], 2) + pow(z[i + 1] - fnm1 * x[i + 1], 2);
                    }
                    error = sqrt(fnm1 * error);

                    // printf("  %12d  %8d  %12e", n, nits, error);
                    //now, nothing is printed here, this task is made for master process

                    //send result to master process
                    MPI_Send(&n, 1, MPI_INT, dest, TAG_N, MPI_COMM_WORLD);
                    MPI_Send(&nits, 1, MPI_INT, dest, TAG_NITS, MPI_COMM_WORLD);
                    MPI_Send(&error, 1, MPI_DOUBLE, dest, TAG_ERROR, MPI_COMM_WORLD);
                  
                    first = 0;
                }
                else
                {
                    dest = 0;
                    source = 0;
                    ctime1 = cpu_time();
                    for (it = 0; it < nits; it++)
                    {
                        sgn = +1.0;
                        cfft2(n, x, y, w, sgn);
                        sgn = -1.0;
                        cfft2(n, y, x, w, sgn);
                    }
                    ctime2 = cpu_time();
                    ctime = ctime2 - ctime1;

                    flops = 2.0 * (double)nits * (5.0 * (double)n * (double)ln2);

                    mflops = flops / 1.0E+06 / ctime;

                    // printf("  %12e  %12e  %12f\n", ctime, ctime / (double)(2 * nits), mflops);   
                    //now, nothing is printed here, this task is made for master process

                    //send result to master process
                    MPI_Send(&ctime, 1, MPI_DOUBLE, dest, TAG_TIME, MPI_COMM_WORLD);
                    MPI_Send(&nits, 1, MPI_INT, dest, TAG_NITS2, MPI_COMM_WORLD);
                    MPI_Send(&mflops, 1, MPI_DOUBLE, dest, TAG_MFLOPS, MPI_COMM_WORLD);
                }
            }
            if ((ln2 % 4) == 0)
            {
                nits = nits / 10;
            }
            if (nits < 1)
            {
                nits = 1;
            }

            free(w);
            free(x);
            free(y);
            free(z);
        }
        }
        
        
        inicio = total/rank*r;
   
      }

    
    }

    //finalized time execution mpi
    endtime = MPI_Wtime();
    MPI_Finalize();
         if ( rank == MASTER)//just master made this task
    {
        printf("\n");
        printf("FFT_MPI:\n");
        printf("  Normal end of execution.\n");
        printf("\n");
        printf("That took %f seconds\n",endtime-starttime);
    }
    return 0;
}


/******************************************************************************/

void ccopy(int n, double x[], double y[])

/******************************************************************************/
/*
Purpose:

  CCOPY copies a complex vector.

Discussion:

  The "complex" vector A[N] is actually stored as a double vector B[2*N].

  The "complex" vector entry A[I] is stored as:

    B[I*2+0], the real part,
    B[I*2+1], the imaginary part.

Modified:

  23 March 2009

Author:

  Original C version by Wesley Petersen.
  This C version by John Burkardt.

Reference:

  Wesley Petersen, Peter Arbenz,
  Introduction to Parallel Computing - A practical guide with examples in C,
  Oxford University Press,
  ISBN: 0-19-851576-6,
  LC: QA76.58.P47.

Parameters:

  Input, int N, the length of the vector.

  Input, double X[2*N], the vector to be copied.

  Output, double Y[2*N], a copy of X.
*/
{
    int i;

    for (i = 0; i < n; i++)
    {
        y[i * 2 + 0] = x[i * 2 + 0];
        y[i * 2 + 1] = x[i * 2 + 1];
    }
    return;
}
/******************************************************************************/

void cfft2(int n, double x[], double y[], double w[], double sgn)

/******************************************************************************/
/*
Purpose:

  CFFT2 performs a complex Fast Fourier Transform.

Modified:

  23 March 2009

Author:

  Original C version by Wesley Petersen.
  This C version by John Burkardt.

Reference:

  Wesley Petersen, Peter Arbenz,
  Introduction to Parallel Computing - A practical guide with examples in C,
  Oxford University Press,
  ISBN: 0-19-851576-6,
  LC: QA76.58.P47.

Parameters:

  Input, int N, the size of the array to be transformed.

  Input/output, double X[2*N], the data to be transformed.
  On output, the contents of X have been overwritten by work information.

  Output, double Y[2*N], the forward or backward FFT of X.

  Input, double W[N], a table of sines and cosines.

  Input, double SGN, is +1 for a "forward" FFT and -1 for a "backward" FFT.
*/
{
    int j;
    int m;
    int mj;
    int tgle;

    m = (int)(log((double)n) / log(1.99));
    mj = 1;
    /*
  Toggling switch for work array.
*/
    tgle = 1;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    if (n == 2)
    {
        return;
    }

    for (j = 0; j < m - 2; j++)
    {
        mj = mj * 2;
        if (tgle)
        {
            step(n, mj, &y[0 * 2 + 0], &y[(n / 2) * 2 + 0], &x[0 * 2 + 0], &x[mj * 2 + 0], w, sgn);
            tgle = 0;
        }
        else
        {
            step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);
            tgle = 1;
        }
    }
    /*
  Last pass through data: move Y to X if needed.
*/
    if (tgle)
    {
        ccopy(n, y, x);
    }

    mj = n / 2;
    step(n, mj, &x[0 * 2 + 0], &x[(n / 2) * 2 + 0], &y[0 * 2 + 0], &y[mj * 2 + 0], w, sgn);

    return;
}
/******************************************************************************/

void cffti(int n, double w[])

/******************************************************************************/
/*
Purpose:

  CFFTI sets up sine and cosine tables needed for the FFT calculation.

Modified:

  23 March 2009

Author:

  Original C version by Wesley Petersen.
  This C version by John Burkardt.

Reference:

  Wesley Petersen, Peter Arbenz,
  Introduction to Parallel Computing - A practical guide with examples in C,
  Oxford University Press,
  ISBN: 0-19-851576-6,
  LC: QA76.58.P47.

Parameters:

  Input, int N, the size of the array to be transformed.

  Output, double W[N], a table of sines and cosines.
*/
{
    double arg;
    double aw;
    int i;
    int n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ((double)n);

    for (i = 0; i < n2; i++)
    {
        arg = aw * ((double)i);
        w[i * 2 + 0] = cos(arg);
        w[i * 2 + 1] = sin(arg);
    }
    return;
}
/*******************************************************************************/

double cpu_time(void)

/*******************************************************************************/
/*
Purpose:

  CPU_TIME reports the total CPU time for a program.

Modified:

  27 September 2005

Author:

  John Burkardt

Parameters:

  Output, double CPU_TIME, the current total elapsed CPU time in second.
*/
{
    double value;

    value = (double)clock() / (double)CLOCKS_PER_SEC;

    return value;
}
/******************************************************************************/

double ggl(double *seed)

/******************************************************************************/
/*
Purpose:

  GGL generates uniformly distributed pseudorandom real numbers in [0,1].

Modified:

  23 March 2009

Author:

  Original C version by Wesley Petersen, M Troyer, I Vattulainen.
  This C version by John Burkardt.

Reference:

  Wesley Petersen, Peter Arbenz,
  Introduction to Parallel Computing - A practical guide with examples in C,
  Oxford University Press,
  ISBN: 0-19-851576-6,
  LC: QA76.58.P47.

Parameters:

  Input/output, double *SEED, used as a seed for the sequence.

  Output, double GGL, the next pseudorandom value.
*/
{
    double d2 = 0.2147483647e10;
    double t;
    double value;

    t = *seed;
    t = fmod(16807.0 * t, d2);
    *seed = t;
    value = (t - 1.0) / (d2 - 1.0);

    return value;
}
/******************************************************************************/

void step(int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn)

    /******************************************************************************/
    /*
    Purpose:

      STEP carries out one step of the workspace version of CFFT2.

    Modified:

      23 March 2009

    Author:

      Original C version by Wesley Petersen.
      This C version by John Burkardt.

    Reference:

      Wesley Petersen, Peter Arbenz,
      Introduction to Parallel Computing - A practical guide with examples in C,
      Oxford University Press,
      ISBN: 0-19-851576-6,
      LC: QA76.58.P47.

    Parameters:
  */
{
    double ambr;
    double ambu;
    int j;
    int ja;
    int jb;
    int jc;
    int jd;
    int jw;
    int k;
    int lj;
    int mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    for (j = 0; j < lj; j++)
    {
        jw = j * mj;
        ja = jw;
        jb = ja;
        jc = j * mj2;
        jd = jc;

        wjw[0] = w[jw * 2 + 0];
        wjw[1] = w[jw * 2 + 1];

        if (sgn < 0.0)
        {
            wjw[1] = -wjw[1];
        }

        for (k = 0; k < mj; k++)
        {
            c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0];
            c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1];

            ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0];
            ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1];

            d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }
    return;
}
/******************************************************************************/

void timestamp()

/******************************************************************************/
/*
Purpose:

  TIMESTAMP prints the current YMDHMS date as a time stamp.

Example:

  31 May 2001 09:45:54 AM

Licensing:

  This code is distributed under the GNU LGPL license.

Modified:

  24 September 2003

Author:

  John Burkardt

Parameters:

  None
*/
{
    #define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time(NULL);
    tm = localtime(&now);

    strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

    printf("%s\n", time_buffer);

    return;
    #undef TIME_SIZE
}
