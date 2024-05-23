// Copyright 2023 National Solar Observatory / AURA, https://www.nso.edu
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#include "config.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>

#ifdef HAS_X86_INTRINSICS
#include <x86intrin.h>
#include <xmmintrin.h>
#endif

#ifdef HAS_SETPRIORITY
#include <sys/resource.h>
#endif

#include "matilda.h"

#ifdef BENCHMARKING_BLAS
#include <dlfcn.h>
extern "C"
{
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
void cblas_sgemv( const enum CBLAS_ORDER order, 
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY );
void cblas_dgemv( const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double *A, const int lda,
                 const double *X, const int incX, const double beta,
                 double *Y, const int incY) ;
                 }
typedef float TYPE;
#endif

static int n_loops = 100000;
static int warmup = -100000;
static constexpr int ALIGNMENT = 64;

static constexpr bool save_mvm_files = true;
// Various delays for timing testing. Enable only one.
static constexpr bool delay_after_warmup = false;
static constexpr bool fixed_delay_spin_loop = false;
static constexpr bool fixed_delay_stay_busy = false;
static constexpr bool sleep_and_wake = false;

static float * init_matrix( int rows, int cols ); // Allocate matrix and fill with some data.
static float * init_input_vector( int cols );     // Allocate vector and fill with some data.
static float * init_output_vector( int rows );    // Allocate vector.
static bool verify_result( float const * matrix, int rows, int cols, float const * vector, float const * other_result, double const epsilon = 1e-5 );
static void spin_wait_ns( uint64_t ns );

extern int mkl_get_num_threads();

int main( int argc, char **argv )
{
#ifdef HAS_X86_INTRINSICS
  _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
  _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
#endif
  mlockall( MCL_FUTURE );

  Matilda::mvm_param p;
  p.n_rows = 3584;
  p.n_cols = 3456;
  p.f16c = false;
#ifdef HAS_AVX2
  p.kernel = Matilda::MvmKernel::AVX2_56;
#else
  p.kernel = Matilda::MvmKernel::OMP_SIMD_56;
#endif
  p.n_threads = 64;
  p.first_cpu = 64;
#ifdef HAS_SCHED_SETAFFINITY
  p.sched_policy = SCHED_FIFO;
  p.sched_prio = 10;
#endif
#ifdef HAS_SETPRIORITY
  p.nice = -19;
#endif

  long wait = 500'000;
  char timings_file_name[256];
  timings_file_name[0] = '\0';
  int opt;
  while(( opt = getopt( argc, argv, "r:c:t:p:l:w:s:o:k:P:hd" ))  != -1 )
  {
    switch( opt )
    {
      case 'r':
        p.n_rows = atoi( optarg );
        break;
      case 'c':
        p.n_cols = atoi( optarg );
        break;
      case 't':
        p.n_threads = atoi( optarg );
        break;
      case 'p':
        p.first_cpu = atoi( optarg );
        break;
      case 'l':
        n_loops = atoi( optarg );
        break;
      case 'w':
        warmup = - abs( atoi( optarg ) );
        break;
      case 's':
        wait = atol( optarg ) * 1000L;
        break;
      case 'o':
        snprintf( timings_file_name, sizeof(timings_file_name), "%s", optarg );
        break;
      case 'k':
      {
        int const kernel = atoi( optarg );
        p.kernel = static_cast<Matilda::MvmKernel>( kernel );
        break;
      }
      case 'P':
        p.sched_prio = atoi( optarg );
        break;
      case 'd':
        p.f16c = true;
        break;
      case 'h':
      default:
        std::cerr << "Usage: matilda_test [-r number_of_rows] [-c number_of_colums] [-d] [-l number_loop_repetitions] [-w number_of_warmup_loops] [-o timings_file]";
        if constexpr( fixed_delay_spin_loop or fixed_delay_stay_busy )
          std::cerr << " [-s us_to_sleep_between_multiplications]";
#ifndef BENCHMARKING_BLAS
        std::cerr << " [-k mvm_kernel] [-t number_of_threads] [-p index_of_first_processor] [-P scheduler_priority]";
        std::cerr << "\n\navailable mvm_kernels:\n";
        std::cerr << "Chunks of 8 rows:     " << int(Matilda::MvmKernel::OMP_SIMD_8) << " (OMP_SIMD-8)\n";
        std::cerr << "Chunks of 16 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_16) << " (OMP_SIMD-16)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_16) << " (AVX2-16), " << int(Matilda::MvmKernel::AVX2_16_A) << " (AVX2-16a), " << int(Matilda::MvmKernel::AVX2_16_B) << " (AVX2-16b)";
#endif
#ifdef HAS_AVX512F
        std::cerr << ", " << int(Matilda::MvmKernel::AVX512_16) << " (AVX512-16)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 24 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_24) << " (OMP_SIMD-24)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_24) << " (AVX2-24)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 32 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_32) << " (OMP_SIMD-32)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_32) << " (AVX2-32)";
#endif
#ifdef HAS_AVX512F
        std::cerr << ", " << int(Matilda::MvmKernel::AVX512_32) << " (AVX512-32)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 40 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_40) << " (OMP_SIMD-40)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_40) << " (AVX2-40)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 48 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_48) << " (OMP_SIMD-48)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_48) << " (AVX2-48)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 56 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_56) << " (OMP_SIMD-56)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_56) << " (AVX2-56)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 64 rows:   " << int(Matilda::MvmKernel::OMP_SIMD_64) << " (OMP_SIMD-64)";
#ifdef HAS_AVX2
        std::cerr << ", " << int(Matilda::MvmKernel::AVX2_64) << " (AVX2-64)";
#endif
#ifdef HAS_AVX512F
        std::cerr << ", " << int(Matilda::MvmKernel::AVX512_64) << " (AVX512-64)";
#endif
        std::cerr << "\n",

        std::cerr << "Chunks of 128 rows: " << int(Matilda::MvmKernel::OMP_SIMD_128) << " (OMP_SIMD-128)";
        std::cerr << "\n",

        std::cerr << "Chunks of 192 rows: " << int(Matilda::MvmKernel::OMP_SIMD_192) << " (OMP_SIMD-192)";
        std::cerr << "\n",
#endif
        std::cerr << "\n";
        exit( EXIT_FAILURE );
        __builtin_unreachable();
    }
  }

  if( strlen( timings_file_name ) == 0 )
  {
#ifndef BENCHMARKING_BLAS
    snprintf( timings_file_name, sizeof(timings_file_name), "%s_timings_us_r%d_c%d_k%d_t%d_p%d.dat", "matilda", p.n_rows, p.n_cols, p.kernel, p.n_threads, p.first_cpu );
#else
    int (* get_num_threads)() = reinterpret_cast<int(*)()>( dlsym( RTLD_DEFAULT, "openblas_get_num_threads" ) );
    if( get_num_threads == nullptr )
      get_num_threads = reinterpret_cast<int(*)()>( dlsym( RTLD_DEFAULT, "mkl_get_max_threads" ) );
    int n_threads =  (*get_num_threads)();
    snprintf( timings_file_name, sizeof(timings_file_name), "%s_timings_us_r%d_c%d_t%d.dat", BLAS_PROVIDER, p.n_rows, p.n_cols, n_threads );
#endif
  }

  std::cout << "Computing " << p.n_rows << "x" << p.n_cols << " MVM " << n_loops << " times (+ " << std::abs(warmup) << " times for warming up)";
  if constexpr( fixed_delay_spin_loop or fixed_delay_stay_busy )
    std::cout << " with " << wait/1000 << " micro-seconds in-between";
#ifndef BENCHMARKING_BLAS
  std::cout << ", using " << p.n_threads << " threads starting at CPU " << p.first_cpu;
#endif
  if( p.f16c )
    std::cout << ". Matrix stored with 16-bit half-precision";
  std::cout << ".\n";


#ifndef BENCHMARKING_BLAS
#ifdef HAS_SCHED_SETAFFINITY
  cpu_set_t cps;
  CPU_ZERO( &cps );
  CPU_SET( p.first_cpu-1, &cps );
  if( sched_setaffinity( 0, sizeof(cps), &cps ) != 0 )
    perror( "Matilda - Failed to set CPU affinity" );
#endif
#endif

  p.matrix = init_matrix( p.n_rows, p.n_cols );
  p.vector = init_input_vector( p.n_cols );
  p.result = init_output_vector( p.n_rows );

#ifndef BENCHMARKING_BLAS
  Matilda::mvm_plan plan( p );

#ifdef HAS_SCHED_SETSCHEDULER
  struct sched_param sched_p;
  sched_p.sched_priority = p.sched_prio;
  if( sched_setscheduler( 0, p.sched_policy, &sched_p ) != 0 )
    perror( "Matilda - sched_setscheduler failed" );
#endif
#ifdef HAS_SETPRIORITY
  if( setpriority( PRIO_PROCESS, 0, p.nice ) != 0 )
    perror( "Matilda - setpriority failed" );
#endif
#endif

  // Variables for timing measurements
  float mean = 0;
  float var = 0;
  float mi = std::numeric_limits<float>::max();
  float ma = 0;
  float * timings = new float[abs(warmup)+n_loops+1]; // nanoseconds

  // Start compute loop
  for( int l=warmup; l<=n_loops; l++ )
  {
    uint64_t t0 = __rdtsc();
#ifndef BENCHMARKING_BLAS
    plan.execute();
#else
    static constexpr TYPE alpha = 1;
    static constexpr TYPE beta = 0;
    if constexpr( std::is_same<TYPE,double>::value )
    {
      cblas_dgemv( CblasRowMajor, CblasNoTrans, p.n_rows, p.n_cols, alpha, reinterpret_cast<double*>( p.matrix ), p.n_cols, reinterpret_cast<double*>( p.vector ), 1, beta, reinterpret_cast<double*>( p.result ), 1 );
    }
    else if constexpr( std::is_same<TYPE,float>::value )
    {
      cblas_sgemv( CblasRowMajor, CblasNoTrans, p.n_rows, p.n_cols, alpha, reinterpret_cast<float*>( p.matrix ), p.n_cols, reinterpret_cast<float*>( p.vector ), 1, beta, reinterpret_cast<float*>( p.result ), 1 );
    }
#endif
    uint64_t t1 = __rdtsc();

    //
    // Various sources of delay
    //
    if( l > 0 ) [[likely]]
    {
      // Short delay after warming up
      if constexpr( delay_after_warmup )
      {
        if( l < 20 )
          spin_wait_ns( wait );
      }

      // Fixed delay with spin loop
      if constexpr( fixed_delay_spin_loop )
      {
        spin_wait_ns( wait );
      }

#ifndef BENCHMARKING_BLAS
      // Fixed delay with staying busy
      if constexpr( fixed_delay_stay_busy )
      {
        long pre = 50'000;
        if( wait - pre > 0 )
        {
          plan.stay_busy();
          spin_wait_ns( wait - pre );
          plan.get_ready();
          spin_wait_ns( pre ); // Give threads some time to finish current MVM to avoid that the finishing is measured in the next plan.execute().
        }
      }

      // Sleeping and waking up
      if constexpr( sleep_and_wake )
      {
        if( l == n_loops / 2 )
        {
          plan.sleep();
          spin_wait_ns( 10'000'000'000UL );
          plan.wakeup();
        }
      }
#endif
    }

    // Store timings and compute some statistics
    float const timing_us = static_cast<float>(t1 - t0) * Matilda::mvm_plan::tsc_tic_ns() / 1000.f;
    if( l > 0 ) [[likely]]
    {
      float old_mean = mean;
      mean = mean + ( timing_us - mean ) / l;
      var = var + ( ( timing_us - old_mean ) * ( timing_us - mean ) - var ) / l;
      mi = std::min(mi,timing_us);
      ma = std::max(ma,timing_us);
    }
    timings[l+abs(warmup)] = timing_us;
  }
#ifndef BENCHMARKING_BLAS
  plan.sleep();
#endif
  float gflop = ((2*p.n_rows*p.n_cols-p.n_rows));
  printf( "%.3f +/- %.3f us per %dx%d MVM. (%.0f GFLOP/s), min: %.3f us (%.0f GFLOP/s), max: %.3f us (%.0f GFLOP/s), %.0f\n",
          mean, sqrt(var), p.n_rows, p.n_cols, gflop/mean/1000, mi, gflop/mi/1000, ma, gflop/ma/1000, p.result[0] );

  bool const correct = verify_result( p.matrix, p.n_rows, p.n_cols, p.vector, p.result, p.f16c ? 1e-3:1e-5 );

  if( strlen(timings_file_name) )
  {
    printf("Writing MVM timings (32-bit floats) to %s.\n", timings_file_name );
    FILE * f = fopen( timings_file_name, "w");
    fwrite( static_cast<void*>(timings), sizeof(*timings), abs(warmup)+n_loops+1, f );
    fclose(f);
  }

  free( p.matrix );
  free( p.vector );
  free( p.result );
  delete[] timings;

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}


float * init_matrix( int const rows, int const cols )
{
  float * matrix;
  posix_memalign( reinterpret_cast<void**>( &matrix ), ALIGNMENT, rows * cols * sizeof(float) );
  int n = 0;
  for( int r = 0; r < rows; r++ )
    for( int c = 0; c < cols; c++ )
      //       matrix[c+cols*r] = r==c ? 1 : 0;
      //       matrix[c+cols*r] = n++;
      matrix[c+cols*r] = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX ) + 0.5f;
  return matrix;
}

float * init_input_vector( int const cols )
{
  float * vec;
  posix_memalign( reinterpret_cast<void**>( &vec ), ALIGNMENT, cols * sizeof(float) );

  for( int c = 0; c < cols; c++ )
    vec[c] = static_cast<float>( rand() ) / static_cast<float>( RAND_MAX ) + 0.5f;
  return vec;
}

float * init_output_vector( int const rows )
{
  float * vec;
  posix_memalign( reinterpret_cast<void**>( &vec ), ALIGNMENT, rows * sizeof(float) );
  for( int c = 0; c < rows; c++ )
    vec[c] = 0;
  return vec;
}

bool verify_result( float const * matrix, int const rows, int const cols, float const * vector, float const * other_result, double const epsilon )
{
  if constexpr( save_mvm_files )
  {
    FILE * f = fopen( "matilda_vector.dat", "w" );
    fwrite( static_cast<void const*>( vector ), sizeof(float), cols, f );
    fclose(f);

    f = fopen( "matilda_matrix.dat", "w" );
    fwrite( static_cast<void const*>( matrix ), sizeof(float), rows*cols, f );
    fclose(f);

    f = fopen( "matilda_result.dat", "w" );
    fwrite( static_cast<void const*>( other_result ), sizeof(float), rows, f );
    fclose(f);
  }

  double * result = new double[rows];
  memset( result, 0, sizeof(*result)*rows );
  for( int r = 0; r < rows; r++ )
    for( int c = 0; c < cols; c++ )
      result[r] += static_cast<double>( matrix[r*cols+c] ) * static_cast<double>( vector[c] );

  bool all_correct = true;
  for( int r = 0; r < rows; r++ )
  {
    if( std::abs( result[r] - static_cast<double>( other_result[r] ) ) >  epsilon * std::abs( result[r] ) )
    {
      printf("!!!!!!!!\n!!!!!!!! Result element %d incorrect: should be %f, but is %f\n!!!!!!!!\n", r, result[r], other_result[r] );
      all_correct = false;
      break;
    }
  }
  if( all_correct )
    printf("MVM correct within %.1e\n", epsilon );
  delete[] result;

  return all_correct;
}

void spin_wait_ns( uint64_t ns )
{
  static float const inv_tsc_period_ns = 1.f / Matilda::mvm_plan::tsc_tic_ns();
  uint64_t const tic = __rdtsc();
  uint64_t const diff =  static_cast<uint64_t>( static_cast<float>( ns ) * inv_tsc_period_ns );
  while( __rdtsc() - tic < diff )
  {
#ifdef HAS_X86_INTRINSICS
    for( int i = 0; i < 15; i++ ) // This number seems to minimize the thread's latency to changes of the task on AMD 7742
      _mm_pause();
#endif
  }
}
