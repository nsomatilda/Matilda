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

#pragma once


#include <atomic>
#include <condition_variable>
#include <mutex>


namespace Matilda
{
  enum class ThreadTask : int { Initialize, ComputeMVM, SpinWait, StayBusy, Sleep, Terminate, Dead };
  enum class MvmKernel : int {
#ifdef HAS_AVX2
    AVX2_16=25616, AVX2_16_A=256161, AVX2_16_B=256162, AVX2_32=25632, AVX2_40=25640, AVX2_48=25648, AVX2_56=25656, AVX2_64=25664,
#endif
#ifdef HAS_AVX512F
  AVX512_16=51216, AVX512_32=51232, AVX512_64=51264,
#endif
  OMP_SIMD_8=8, OMP_SIMD_16=16, OMP_SIMD_24=24, OMP_SIMD_32=32, OMP_SIMD_40=40, OMP_SIMD_48=48, OMP_SIMD_56=56, OMP_SIMD_64=64, OMP_SIMD_128=128, OMP_SIMD_192=192};

  struct mvm_param
  {
    float * matrix;      ///< Pointer to the row-major ordered, contiguous matrix used for the MVM.
    float * vector;      ///< Pointer to the input vector used for the MVM. Must be aligned on 64-byte boundaries.
    float * result;      ///< Pointer to the output vector of the MVM. Must be aligned on 64-byte boundaries.
    int n_rows;          ///< Number of rows in the matrix and number of elements in the result vector.
    int n_cols;          ///< Number of columns in the matrix and number of elements in the input vector.
    MvmKernel kernel;    ///< Processing kernel used for the MVM.
    int n_threads;       ///< Number of parallel threads for the multiplication. These threads will be created in addition to the calling thread.
                         ///< The calling thread syncronizes the threads but does is not involved with tha parallel matrix-vector multiplication.
    int first_cpu;       ///< Index of the CPU used by the first thread. All other threads will be contiguous on subsequent CPUs.
    int sched_policy;    ///< Operating system scheduler policy for the threads.
    int sched_prio;      ///< Operating system schedulder priority for the threads.
    int nice;            ///< Operating system schedulder priority ("nice level") for the threads.
    bool f16c;           ///< True if the matrix data is stored in half-precision format. Input and output vectors are stored with single precision.
  };


  class mvm_plan
  {
    public:
      ///< Constructor of an mvm_plan.
      mvm_plan( const mvm_param & p );

      // Do not accidentally create or copy an object instance.
      mvm_plan() = delete;
      mvm_plan( const mvm_plan & plan ) = delete;
      mvm_plan& operator=( const mvm_plan & plan ) = delete;

      ///< Destructor of an mvm_plan.
      ~mvm_plan();

      ///< Perform the matrix-vector multiplication.
      void execute();

      ///< Put the worker threads to sleep until woken up by mvm_plan::wakeup(). This is useful relieve the CPUs when repeated multiplications are not needed.
      ///< If not in sleep, the worker threads will use 100% CPU time either computing MVMs or busy spinning.
      void sleep();

      ///< Wake up the worker threads after mvm_plan::sleep() was used to put them to sleep.
      void wakeup();

      ///< Execute the MVM repeatedly to keep the CPU without storing the result until interrupted by mvm_plan::get_ready().
      ///< This function is a workaround for certain processors (such as AMD Epyc 7002) that slow down despite the busy loop. 
      void stay_busy();

      ///< Stop repeated MVM started by mvm_plan::stay_busy(). The ongoing MVM will complete before the next mvm_plan::execute() will be processed. To minimize latency, call mvm_plan::get_ready() early enough.
      void get_ready(){m_stay_busy = false;}

      ///< Returns the length of time stamp counter increment in nanoseconds.
      static float tsc_tic_ns() noexcept;

    private:
      // Functions executed in the main thread
      void check_arguments( const mvm_param & p ) const;
      void start_and_initialize_worker_threads();
      void do_parallel( ThreadTask new_task ) noexcept;
      void join_parallel() const noexcept;

      // Functions executed in parallel by the worker threads
      void thread_impl( int threadnr );
      void thread_impl_compute_mvm( int threadnr );
      void thread_impl_initialize( int threadnr );
      void thread_impl_sleep( int threadnr );
      void thread_impl_stay_busy( int threadnr ) noexcept;
      void thread_impl_terminate( int threadnr );
      void (*mvm_kernel_func)( float const * matrix, float const * vector, size_t width, float * result ); // Pointer to the MVM kernel function that will be executed for the MVM.

      // Functions executed in any thread
      void spin_wait_ns( int ns ) const noexcept;

      std::atomic<ThreadTask> * m_task; // Array of task setters for the threads. Size: m_num_threads

      float *** m_localmatrix;          // Pointer to an array of matrix chunks (each matrix chunk is contiguous), one per thread. [threadnr][chunknr][elementnr].
                                        // Each chunk is column-major copy of the corresponding rows in origmatrix. Chunks are allocated by the respective thread.
      float * m_vector;                 // Pointer to the input vector for the multiplication. Externally allocated. Must be aligned on 64-byte boundaries.
      float * m_result;                 // Pointer to the result vector for the multiplication. Externally allocated. Must be aligned on 64-byte boundaries.
      float ** m_localresult;           // Pointer to the result vector for the multiplication. Internally allocated by the respective thread.

      int const m_num_cols;             // Number of columns in the matrix and number of elements in the input vector.
      int const m_num_rows;             // Number of rows in the matrix and number of elements in the result vector.
      int const m_num_threads;          // Number of worker threads that compute the MVM. The main thread does not compute parts of the MVM.
      int m_simd_rows;                  // Number of rows processed in the selected MVM kernel.

#ifdef MATILDA_BENCHMARK_THREADS
      float ** m_timings;
      size_t const m_max_num_benchmark;
#endif

      float const * m_origmatrix;       // Pointer to the original row-major ordered, contiguous matrix
      float * m_busy_vector;            // Pointer to the input vector for the multiplication used when staying busy.
      bool m_f16c;                      // Use F16C format to store matrix data?
      MvmKernel m_kernel;               // Kernel used in the MVM.
      int const m_first_cpu;            // Index of the CPU used by the first thread. All other threads will be contiguous on subsequent CPUs.
      int const m_sched_policy;         // Operating system scheduler policy for the threads.
      int const m_sched_prio;           // Operating system schedulder priority for the threads.
      int const m_nice;                 // Operating system schedulder priority ("nice level") for the threads.
      static long const m_page_size;    // Page size for this process.

      std::atomic<bool> m_stay_busy;    // Keep threads computing dummy MVMs between execute() calls?

      std::atomic<bool> m_sleep;        // Put threads to sleep?
      std::condition_variable m_cv;     //
      std::mutex m_cv_mutex;            //
    };
}

