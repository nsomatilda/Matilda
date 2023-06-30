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
#include "version.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <unistd.h>

#ifdef HAS_X86_INTRINSICS
#include <x86intrin.h>
#endif

#ifdef HAS_SCHED_SETAFFINITY
#include <sched.h>
#endif

#ifdef HAS_SETPRIORITY
#include <sys/resource.h>
#endif

#ifdef MATILDA_USE_PERF
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#endif


#include "matilda.h"
#ifdef HAS_AVX2
#include "mvm_impl_avx2.h"
#endif
#ifdef HAS_AVX512F
#include "mvm_impl_avx512.h"
#endif
#include "mvm_impl_omp_simd.h"

using namespace Matilda;


static float get_tsc_period_ns();

static constexpr bool VERBOSE_TSC = false;
static constexpr int ALIGNMENT = 64;
static constexpr int CACHE_LINE = 64;
static constexpr int MAX_STACK_SIZE = 4*1024*1204;
static constexpr int TASKS_PER_CACHE_LINE = 4; // 4 is fastest (~ 2 us overhead for 64 threads), 1 is slowest (slower than 16) on AMD 7742.

static_assert( sizeof( std::atomic<ThreadTask> ) <= CACHE_LINE );
static_assert( std::atomic<ThreadTask>::is_always_lock_free );

static float g_tsc_period_ns = get_tsc_period_ns();
long const mvm_plan::m_page_size = sysconf( _SC_PAGESIZE );

thread_local int tl_n_chunks = 0;  // number of matrix chunks per thread (currently the same for all threads)
#ifdef MATILDA_BENCHMARK_THREADS
thread_local size_t tl_benchmark_counter = 0;
#endif


mvm_plan::mvm_plan( const mvm_param & p ) :
    m_vector( p.vector ),
    m_result( p.result ),
    m_num_cols( p.n_cols ),
    m_num_rows( p.n_rows ),
    m_num_threads( p.n_threads ),
#ifdef MATILDA_BENCHMARK_THREADS
    m_max_num_benchmark( 2000000 ),
#endif
    m_origmatrix( p.matrix),
    m_f16c( p.f16c ),
    m_kernel( p.kernel ),
    m_first_cpu( p.first_cpu ),
    m_sched_policy( p.sched_policy ),
    m_sched_prio( p.sched_prio ),
    m_nice( p.nice ),
    m_stay_busy( false ),
    m_sleep( false )
{
#ifdef MATILDA_BENCHMARK_THREADS
  std::cerr << "Matilda - Benchmarking enabled, " << g_tsc_period_ns << " nanoseconds per TSC tic, (" << 1./g_tsc_period_ns << " GHz)\n";
#endif

  switch( m_kernel )
  {
#ifdef HAS_AVX2
    case MvmKernel::AVX2_16:
        m_simd_rows = 16;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_16_f16c :  &mvm_kernel_avx2_16;
        break;
    case MvmKernel::AVX2_16_A:
        m_simd_rows = 16;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_16_a_f16c : &mvm_kernel_avx2_16_a;
        break;
    case MvmKernel::AVX2_16_B:
        m_simd_rows = 16;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_16_b_f16c : &mvm_kernel_avx2_16_b;
        break;
    case MvmKernel::AVX2_32:
        m_simd_rows = 32;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_32_f16c : &mvm_kernel_avx2_32;
        break;
    case MvmKernel::AVX2_40:
        m_simd_rows = 40;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_40_f16c : &mvm_kernel_avx2_40;
        break;
    case MvmKernel::AVX2_64:
        m_simd_rows = 64;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_64_f16c : &mvm_kernel_avx2_64;
        break;
    case MvmKernel::AVX2_56:
        m_simd_rows = 56;
        mvm_kernel_func = m_f16c ? &mvm_kernel_avx2_56_f16c : &mvm_kernel_avx2_56;
        break;
#endif
#ifdef HAS_AVX512F
    case MvmKernel::AVX512_16:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when AVX-512 MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 16;
        mvm_kernel_func = &mvm_kernel_avx512_16;
        break;
    case MvmKernel::AVX512_32:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when AVX-512 MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 32;
        mvm_kernel_func = &mvm_kernel_avx512_32;
        break;
    case MvmKernel::AVX512_64:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when AVX-512 MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 64;
        mvm_kernel_func = &mvm_kernel_avx512_64;
        break;
#endif
    case MvmKernel::OMP_SIMD_8:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 8;
        mvm_kernel_func = &mvm_kernel_omp_simd<8>;
        break;
    case MvmKernel::OMP_SIMD_16:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 16;
        mvm_kernel_func = &mvm_kernel_omp_simd<16>;
        break;
    case MvmKernel::OMP_SIMD_24:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 24;
        mvm_kernel_func = &mvm_kernel_omp_simd<24>;
        break;
    case MvmKernel::OMP_SIMD_32:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 32;
        mvm_kernel_func = &mvm_kernel_omp_simd<32>;
        break;
    case MvmKernel::OMP_SIMD_40:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 40;
        mvm_kernel_func = &mvm_kernel_omp_simd<40>;
        break;
    case MvmKernel::OMP_SIMD_48:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 48;
        mvm_kernel_func = &mvm_kernel_omp_simd<48>;
        break;
    case MvmKernel::OMP_SIMD_56:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 56;
        mvm_kernel_func = &mvm_kernel_omp_simd<56>;
        break;
    case MvmKernel::OMP_SIMD_64:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 64;
        mvm_kernel_func = &mvm_kernel_omp_simd<64>;
        break;
    case MvmKernel::OMP_SIMD_128:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 128;
        mvm_kernel_func = &mvm_kernel_omp_simd<128>;
        break;
    case MvmKernel::OMP_SIMD_192:
        if( m_f16c )
        {
          std::string msg = std::string( "Matilda - F16C storage not supported when OMP SIMD MVM kernels are used." );
          throw std::invalid_argument( msg );
        }
        m_simd_rows = 192;
        mvm_kernel_func = &mvm_kernel_omp_simd<192>;
        break;
    default:
        std::string msg = std::string( "Matilda - invalid MVM kernel." );
        throw std::invalid_argument( msg );
  }

  check_arguments( p );


  // Set CPU affinity of primary thread
#ifdef HAS_SCHED_SETAFFINITY
  cpu_set_t cps_default;
  if( sched_getaffinity( 0, sizeof(cps_default), &cps_default ) != 0 )
    perror( "Matilda - Failed to get CPU affinity" );

  cpu_set_t cps;
  CPU_ZERO( &cps );
  CPU_SET( m_first_cpu, &cps );
  if( sched_setaffinity( 0, sizeof(cps), &cps ) != 0 )
    perror( "Matilda - Failed to set CPU affinity" );
#endif

  //
  posix_memalign( reinterpret_cast<void**>( &m_task ), ALIGNMENT, m_num_threads * CACHE_LINE / TASKS_PER_CACHE_LINE );
  for( int i = 0; i < m_num_threads; i++ )
  {
    m_task[i*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] = ThreadTask::SpinWait;
  }

  // Allocate array of local matrix pointers
  posix_memalign( reinterpret_cast<void**>( &m_localmatrix ), ALIGNMENT, m_num_threads * sizeof(float**) );
  for( int i = 0; i < m_num_threads; i++ )
    m_localmatrix[i] = nullptr;

  posix_memalign( reinterpret_cast<void**>( &m_busy_vector ), ALIGNMENT, m_num_cols * sizeof(float) );

  posix_memalign( reinterpret_cast<void**>( &m_localresult ), ALIGNMENT, m_num_threads * sizeof(float*) );
  for( int i = 0; i < m_num_threads; i++ )
    m_localresult[i] = nullptr;

#ifdef MATILDA_BENCHMARK_THREADS
  m_timings = new float*[m_num_threads+1];
  m_timings[0] = new float[m_max_num_benchmark];
  memset( m_timings[0], 0, sizeof(float)*m_max_num_benchmark );
#endif

  // Reset CPU affinity of primary thread
#ifdef HAS_SCHED_SETAFFINITY
  if( sched_setaffinity( 0, sizeof(cps_default), &cps_default ) != 0 )
    perror( "Matilda - Failed to set CPU affinity" );
#endif

  // Set thread affinities, allocate and touch matrix data
  start_and_initialize_worker_threads();
}

void mvm_plan::check_arguments( const mvm_param & p ) const
{
  if( p.n_rows % m_simd_rows != 0 )
  {
    std::string msg = std::string( "Matilda - Number of matrix rows is not multiple of " ) + std::to_string(m_simd_rows) + ".";
    throw std::invalid_argument( msg );
  }

  if( (p.n_rows / m_simd_rows ) % p.n_threads != 0 )
  {
    std::string msg = std::string( "Matilda - Number of matrix rows is not multiple of " ) + std::to_string(m_simd_rows *  p.n_threads) + ".";
    throw std::invalid_argument( msg );
  }

  if( reinterpret_cast<std::uintptr_t>( p.vector ) % ALIGNMENT != 0 )
  {
    std::string msg = std::string( "Matilda - Input vector not aligned to " ) + std::to_string(ALIGNMENT) + " bytes.";
    throw std::invalid_argument( msg );
  }

  if( reinterpret_cast<std::uintptr_t>( p.result ) % ALIGNMENT != 0 )
  {
    std::string msg = std::string( "Matilda - Result vector not aligned to " ) + std::to_string(ALIGNMENT) + " bytes.";
    throw std::invalid_argument( msg );
  }

  if( p.first_cpu < 0 )
  {
    std::string msg = std::string( "Matilda - First CPU for worker threads is negative." );
    throw std::invalid_argument( msg );
  }

  long const num_cpus = sysconf( _SC_NPROCESSORS_ONLN );
  if( p.first_cpu + p.n_threads > num_cpus )
  {
    std::string msg = std::string( "Matilda - Number of worker threads exceeds number of available CPU cores." );
    throw std::invalid_argument( msg );
  }
}

void mvm_plan::execute()
{
#ifdef MATILDA_BENCHMARK_THREADS
  uint64_t t0 = __rdtsc();
#endif

  do_parallel( ThreadTask::ComputeMVM );
  join_parallel();

#ifdef MATILDA_BENCHMARK_THREADS
  uint64_t t1 = __rdtsc();
  float x = static_cast<float>(t1 - t0) * g_tsc_period_ns;
  if( tl_benchmark_counter < m_max_num_benchmark )
    m_timings[0][tl_benchmark_counter++] = x;
#endif
}

void mvm_plan::do_parallel( ThreadTask new_task ) noexcept
{
  for( int i = 0; i < m_num_threads; i++ )
  {
    m_task[i*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] = new_task;
  }
}

void mvm_plan::join_parallel() const noexcept
{
  bool alldone;
  do
  {
    alldone = true;
    for( int i = 0; i < m_num_threads; i++ )
      alldone = alldone && ( m_task[i*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] == ThreadTask::SpinWait );

//    spin_wait_ns( 2000 );
  } while( not alldone );
}


void mvm_plan::spin_wait_ns( int ns ) const noexcept
{
// TODO: non x86 versions
#ifdef HAS_X86_INTRINSICS
  uint64_t const tic = __rdtsc();
  uint64_t const diff =  static_cast<uint64_t>( static_cast<float>( ns ) / g_tsc_period_ns );
  while( __rdtsc() - tic < diff )
  {

    for( int i = 0; i < 15; i++ ) // This number seems to minimize the thread's latency to changes of the task on AMD 7742
      _mm_pause();
  }
#else
#error spin_wait_ns() not implemented for this platform
#endif
}


void mvm_plan::start_and_initialize_worker_threads()
{
  // Launch threads
  for( int i = 0; i < m_num_threads; i++ )
  {
    std::thread worker( &mvm_plan::thread_impl, this, i );
    worker.detach();
  }

  do_parallel( ThreadTask::Initialize );

  join_parallel();
}

void mvm_plan::stay_busy()
{
  m_stay_busy = true;
  do_parallel( ThreadTask::StayBusy );
}

void Matilda::mvm_plan::sleep()
{
  m_stay_busy = false;
  std::unique_lock<std::mutex> lock( m_cv_mutex );
  m_sleep = true;
  lock.unlock();
  do_parallel( ThreadTask::Sleep );
}

void mvm_plan::thread_impl( int threadnr )
{
  bool keepalive = true;
  do
  {
    switch( m_task[threadnr*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] )
    {
      case ThreadTask::ComputeMVM:
            thread_impl_compute_mvm( threadnr );
        break;
      case ThreadTask::SpinWait:
        spin_wait_ns( 500 );
        break;
      case ThreadTask::StayBusy:
            thread_impl_stay_busy( threadnr );
        break;
      case ThreadTask::Initialize:
            thread_impl_initialize( threadnr );
        break;
      case ThreadTask::Sleep:
            thread_impl_sleep( threadnr );
        break;
      case ThreadTask::Terminate:
            thread_impl_terminate( threadnr );
        keepalive = false;
        break;
      case ThreadTask::Dead:
        __builtin_unreachable(); // ThreadTask::Dead is only set in terminate_thread_impl() above and this loop will be left thereafter by the "break".
    }
  } while( keepalive );
}

void mvm_plan::thread_impl_compute_mvm ( int threadnr )
{
#ifdef MATILDA_BENCHMARK_THREADS
  uint64_t t0 = __rdtsc();
#endif

  thread_local bool first = true;
  if( m_sched_prio > 0 )
  {
    if( first ) [[unlikely]]
    {
#ifdef HAS_SCHED_SETSCHEDULER
      struct sched_param sched_p;
      sched_p.sched_priority = m_sched_prio;
      if( sched_setscheduler( 0, m_sched_policy, &sched_p ) != 0 )
        perror( "Matilda - sched_setscheduler failed" );
#endif
      first = false;
    }
  }

  for( int chunknr = 0; chunknr < tl_n_chunks; chunknr++ )
  {
#ifdef MATILDA_USE_LOCAL_RESULT_VECTOR
    float * const result = m_localresult[threadnr] + m_simd_rows * chunknr;
#else
    float * const result = m_result + threadnr * tl_n_chunks * m_simd_rows + m_simd_rows * chunknr;
#endif
    (*mvm_kernel_func )( m_localmatrix[threadnr][chunknr], m_vector, m_num_cols, result );
  }

#ifdef MATILDA_USE_LOCAL_RESULT_VECTOR
  int const c = threadnr*m_simd_rows;
  #pragma omp simd
  for( int i = 0; i < m_simd_rows; i++ )
    m_result[i+c] = m_localresult[threadnr][i];
#endif

#ifdef MATILDA_BENCHMARK_THREADS
  uint64_t t1 = __rdtsc();
  float x = static_cast<float>(t1 - t0) * g_tsc_period_ns;
  if( tl_benchmark_counter < m_max_num_benchmark )
    m_timings[threadnr+1][tl_benchmark_counter++] = x; 
#endif

  m_task[threadnr*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] = ThreadTask::SpinWait;
}

void mvm_plan::thread_impl_initialize( int threadnr )
{
#ifdef HAS_PTHREAD_SETNAME_NP
  char s[16];
  snprintf( s, sizeof(s), "matilda_mvm-%d", threadnr );
#ifndef __APPLE__
  pthread_setname_np( pthread_self(), s );
#else
  pthread_setname_np( s );
#endif
#endif

  // First set the CPU affinity for this thread before any memory is allocated and touched to ensure locality.
#ifdef HAS_SCHED_SETAFFINITY 
  cpu_set_t cps;
  CPU_ZERO( &cps);
  CPU_SET( m_first_cpu + threadnr, &cps );
  if( sched_setaffinity( 0, sizeof(cps), &cps ) != 0 )
    perror( "Matilda - sched_setaffinity failed" );
#endif

  // Pre-fault stack
  volatile char stack[MAX_STACK_SIZE];
  memset( const_cast<char*>(stack), 0, MAX_STACK_SIZE );

#ifdef HAS_SETPRIORITY
  if( setpriority( PRIO_PROCESS, 0, m_nice ) != 0 )
    perror( "Matilda - setpriority failed" );
#endif

  // Allocate local matrix chunks of size simd_rows * cols
  tl_n_chunks = m_num_rows / m_simd_rows / m_num_threads;

  int const num = m_num_cols * m_simd_rows / (m_f16c ? 2 : 1);

  if( threadnr == 0 )
  {
    fprintf( stderr, "Using %d %dx%d matrix block(s) per thread (%lu kiB)\n", tl_n_chunks, m_simd_rows, m_num_cols, tl_n_chunks * num * sizeof(float) / 1024 );
  }
  posix_memalign( reinterpret_cast<void**>( &m_localmatrix[threadnr] ), m_page_size, tl_n_chunks*sizeof(float*) );

  // Copy original matrix data into local chunks
  for( int chunknr = 0; chunknr < tl_n_chunks; chunknr++ )
  {
    posix_memalign( reinterpret_cast<void**>( &m_localmatrix[threadnr][chunknr] ), m_page_size, num*sizeof(float) );

    if( not m_f16c )
    {
      int i = 0;
      for( int c = 0; c < m_num_cols; c++ )
        for( int r = 0; r < m_simd_rows; r++ )
        {
          size_t idx = (threadnr * tl_n_chunks * m_simd_rows + m_simd_rows * chunknr + r) * m_num_cols + c;
          assert( idx < m_num_cols*m_num_rows );
          assert( i < num );
          m_localmatrix[threadnr][chunknr][i++] = m_origmatrix[idx];
        }
    }
    else
    {
#ifdef HAS_AVX2
      int i = 0;
      for( int c = 0; c < m_num_cols; c++ )
        for( int r = 0; r < m_simd_rows; r+=8 )
        {
          size_t idx = (threadnr * tl_n_chunks * m_simd_rows + m_simd_rows * chunknr + r) * m_num_cols + c;
          float const * op = &m_origmatrix[idx];
          __m256 x = _mm256_set_ps( *(op+7*m_num_cols), *(op+6*m_num_cols), *(op+5*m_num_cols), *(op+4*m_num_cols), *(op+3*m_num_cols), *(op+2*m_num_cols), *(op+m_num_cols), *op );
          __m128i d = _mm256_cvtps_ph ( x, _MM_FROUND_TO_NEAREST_INT ); // convert to 8 16-bit floats
          _mm_store_si128( reinterpret_cast<__m128i *>( &m_localmatrix[threadnr][chunknr][i] ), d );
          i += 4; // 4 instead of 8 because of 16-bit float
        }
#endif
    }

//     char filename[128];
//     snprintf( filename, sizeof(filename), "matilda_%d_chunk_%d.dat", threadnr, chunknr ); 
//     FILE * f = fopen( filename, "w" );
//     fwrite( static_cast<void*>( m_localmatrix[threadnr][chunknr] ), sizeof(float), m_num_cols * m_simd_rows, f );
//     fclose(f);

    posix_memalign( reinterpret_cast<void**>( &m_localresult[threadnr] ), m_page_size, m_simd_rows*sizeof(float*) );
  }

#ifdef MATILDA_BENCHMARK_THREADS
  m_timings[threadnr+1] = new float[m_max_num_benchmark];
  memset( m_timings[threadnr+1], 0, sizeof(float)*m_max_num_benchmark );
#endif

  m_task[threadnr*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] = ThreadTask::SpinWait;
}

void Matilda::mvm_plan::thread_impl_sleep ( int threadnr )
{
  std::unique_lock<std::mutex> lock( m_cv_mutex );
  m_cv.wait( lock, [this]{return not m_sleep;} );
}

void mvm_plan::thread_impl_stay_busy( int threadnr ) noexcept
{
  while( m_stay_busy )
  {
    for( int chunknr = 0; chunknr < tl_n_chunks; chunknr++ )
    {
      float * const result = m_localresult[threadnr] + m_simd_rows * chunknr;
      (*mvm_kernel_func)( m_localmatrix[threadnr][chunknr], m_busy_vector, m_num_cols, result );
    }
  }
}

void mvm_plan::thread_impl_terminate( int threadnr )
{
  for( int chunknr = 0; chunknr < tl_n_chunks; chunknr++ )
  {
    free( m_localmatrix[threadnr][chunknr] );
    m_localmatrix[threadnr][chunknr] = nullptr;
  }

  free( m_localmatrix[threadnr] );
  m_localmatrix[threadnr] = nullptr;

  free( m_localresult[threadnr] );
  m_localresult[threadnr] = nullptr;

  m_task[threadnr*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] = ThreadTask::Dead;
}


float Matilda::mvm_plan::tsc_tic_ns() noexcept
{
  return g_tsc_period_ns;
}

void Matilda::mvm_plan::wakeup()
{
  do_parallel( ThreadTask::SpinWait );
  std::unique_lock<std::mutex> lock( m_cv_mutex );
  m_sleep = false;
  lock.unlock();
  m_cv.notify_all();
}

mvm_plan::~mvm_plan()
{
  if( m_sleep )
    wakeup();
  m_stay_busy = false;
  do_parallel( ThreadTask::Terminate );

  bool alldead;
  do
  {
    alldead = true;
    for( int i = 0; i < m_num_threads; i++ )
      alldead = alldead && ( m_task[i*CACHE_LINE / TASKS_PER_CACHE_LINE / sizeof(ThreadTask)] == ThreadTask::Dead );
  } while( not alldead );

  free( m_task );
  free( m_busy_vector );
  free( m_localmatrix );
  free( m_localresult );

  #ifdef MATILDA_BENCHMARK_THREADS
  char filename[256];
  char const * const envname = getenv( "MATILDA_BENCHMARK_OUTPUT" );
  if( envname )
  {
    strncpy( filename, envname, sizeof(filename) );
  }
  else
    strncpy( filename, "/tmp/matilda_benchmark_mvm_timings_nanoseconds.dat", sizeof(filename) );

  FILE * f = fopen( filename, "w");
  if( f )
  {
    bool write_error = false;
    for( int i = 0; i < m_num_threads+1; i++ )
      write_error = write_error or ( fwrite( static_cast<void*>( m_timings[i] ), sizeof(float), tl_benchmark_counter, f ) != tl_benchmark_counter );
    fclose(f);
    if( write_error )
      std::cerr << "Matilda - Error saving timings to " << filename << ".\n";
    else
      std::cerr << "Matilda - Timings in nanoseconds stored in " << filename << ".\n";
  }
  else
  {
    char s[512];
    perror( s );
    snprintf( s, sizeof(s), "Matilda - failed to open file %s", filename );
  }
  #endif
}

static float get_tsc_period_ns()
{
  bool tsc_okay = false;
  float period_ns = 0.f;

  #ifdef MATILDA_ON_LINUX
  // Try Kernel sysfs.
  // Requires kernel module tsc_freq_khz from https://github.com/trailofbits/tsc_freq_khz
  {
    FILE * file = fopen( "/sys/devices/system/cpu/cpu0/tsc_freq_khz", "r" );
    if( file )
    {
      char s[64];
      if( fgets( s, sizeof(s), file ) )
      {
        int tsc_freq_khz = atoi( s );
        if( tsc_freq_khz > 0 )
        {
          period_ns = static_cast<float>( 1e6f / tsc_freq_khz );
          tsc_okay = true;
          if constexpr( VERBOSE_TSC )
            std::cerr << "Matilda - TSC frequency obtained from tsc_freq_khz kernel module.\n";
        }
      }
      fclose( file );
    }
  }
  #endif

  #ifdef MATILDA_USE_PERF
  // Try perf
  // Requires kernel support for perf and privilges
  if( not tsc_okay )
  {
    struct perf_event_attr pe = {
      .type = PERF_TYPE_HARDWARE,
      .size = sizeof(struct perf_event_attr),
      .config = PERF_COUNT_HW_INSTRUCTIONS,
      .disabled = 1,
      .exclude_kernel = 1,
      .exclude_hv = 1
    };

    int fd = static_cast<int>( syscall( __NR_perf_event_open, &pe, 0, -1, -1, 0 ) );
    if( fd == -1 )
    {
      if constexpr( VERBOSE_TSC )
        perror("Matilda - perf_event_open failed");
    }
    else
    {
      size_t const sz = 4*1024;
      void * addr = mmap( NULL, sz, PROT_READ, MAP_SHARED, fd, 0 );
      if( !addr )
      {
        if constexpr( VERBOSE_TSC )
          perror( "Matilda - mmap failed" );
      }
      else
      {
        struct perf_event_mmap_page * pc = static_cast<perf_event_mmap_page*>( addr );
        if( pc->cap_user_time != 1 )
        {
          if constexpr( VERBOSE_TSC )
            std::cerr << "Matilda - Perf system doesn't support user time\n";
        }
        else
        {
          __uint128_t x = 1'000'000'000UL;
          x *= pc->time_mult;
          x >>= pc->time_shift;
          period_ns = x / 1e9f;
          tsc_okay = true;
          if constexpr( VERBOSE_TSC )
            std::cerr << "Matilda - TSC frequency obtained from perf.\n";
        }
        munmap( addr, sz );
      }
      close(fd);
    }
  }
  #endif

  // See if there is an environment variable
  if( not tsc_okay )
  {
    char * s = getenv( "TSC_FREQ_KHZ" );
    if( s )
    {
      int tsc_freq_khz = atoi( s );
      if( tsc_freq_khz > 0 )
      {
        period_ns = static_cast<float>( 1e6f / tsc_freq_khz );
        tsc_okay = true;
        if constexpr( VERBOSE_TSC )
          std::cerr << "Matilda - TSC frequency obtained from TSC_FREQ_KHZ environment variable.\n";
      }
    }
  }


  #ifdef TSC_FREQ_KHZ
  // See if anything is compiled in.
  if( not tsc_okay )
  {
    period_ns = static_cast<float>( 1e6f / TSC_FREQ_KHZ );
    tsc_okay = true;
  }
  #endif

  #ifdef MATILDA_ON_LINUX
  // Last resort, guess from bogomips in /proc/cpuinfo
  if( not tsc_okay )
  {
    char line[256];
    FILE * file = fopen( "/proc/cpuinfo", "r" );
    if( file )
    {
      while( fgets( line, sizeof line, file ) )
      {
        double bogomips = 0.0;
        if( sscanf( line, "bogomips\t: %lf", &bogomips ) == 1 )
        {
          double tsc_freq_Mhz = ( bogomips / 2. ); // WARNING: Factor 1/2 seems valid only for recent AMD and Intel CPUs. https://en.wikipedia.org/wiki/BogoMips
          period_ns = static_cast<float>( 1e3f / tsc_freq_Mhz );
          tsc_okay = true;
          if constexpr( VERBOSE_TSC )
            std::cerr << "Matilda - TSC frequency estimated based on bogomips.\n";
          break;
        }
      }
      fclose( file );
    }
  }
  #endif

  if( not tsc_okay )
  {
    std::string msg = std::string( "Matilda - Time stamp counter frequency could not be determined. Set environment variable TSC_FREQ_KHZ." );
    throw std::invalid_argument( msg );
  }

  return period_ns;
}

