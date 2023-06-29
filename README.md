# Matilda - a library to repeatedly multiply a constant matrix with a variable vector
Matilda is a free C++ library for fast, parallelized matrix-vector multiplications (MVM). 
It is optimized for low latency when repeating the multiplication of a constant matrix with a variable vector over and over. 

Matilda is hosted at GitHub https://github.com/nsomatilda/Matilda

## Features
* Designed for low-latency low-jitter real-time applications
* Fast 32-bit floating point matrix-vector multiplication (MVM)
* Parallelized using its own thread pool designed for low latency
  * Threads are pinned to fixed CPUs and scheduling priorites can be elevated
  * Threads remain active when MVM has finished
* The MVM is executed on dedicated, reserved CPU cores
* Cache-oblivious and NUMA-optimized: matrix data is partitioned, re-ordered for optimal access, and stored to local memory pages
* The calling thread cannot evict matrix data from L1 and L2 caches (and from L3, if not shared between respective CPU cores)
* Explicit, hand-tuned vectorization via AVX2 and AVX-512 intrinsics
* Supports other instruction sets via mandated compiler vectorization
* Supports F16C 16-bit floating point format for matrix data storage to minimize cache use (MVM computed in 32-bit precision)
* Various _kernels_ that compute the MVM for matrices with a row number in multiples 8, 16, 24, 32, 40, 48, 56, and 64 are available to select fastest version for a given matrix size and CPU


## Requirements
Matilda is designed to run on Linux and may or may not work on other operating systems.

For best performance and minimal jitter, the CPU cores dedicated to the MVM should be isolated from other processes and kernel threads such that no context switching and cache pollution occurs.

Matilda can be built with CMake, version 3.14 or higher.

## Compilation
```
mkdir build
cd build
cmake ../
make
make install
```

To use Intel ICC for AMD Zen 2, use `cmake -DCMAKE_CXX_COMPILER=icc -DMARCH="-march=core-avx2 -fma" ../`

## Usage

### Matrix sizing
The number of matrix rows must, in addition to the kernel size, also be devisibe by the product of the number of work threads and the kernel size.
If the size of the matrix intended for the MVM is not compatible with a kernel and the worker thread number, the user must add additional padding rows filled with 0's. 
Experimentation is to find the optimal kernel and thread number is suggested. 

The largest thread number and the next largest kernel seem to be a good starting point.
For example, a 3452x3456 MVM on an AMD 7742 CPU is computed fastest when executed as a 3584x3456 MVM and calculated by 64 threads, i.e. each thread is computing 56 rows using 7 AVX FMA operation per column.


### Test program:
The program `matilda_test` will be compiled but not installed. It can be found in build/apps.
It can be used to benchmark the timing of the MVM and to verify and optimize parameters such as the kernel and the number of threads.
The syntax is
```sh
  matilda_test -r number_of_rows -c number_of_colums -k mvm_kernel -t number_of_threads -p index_of_first_processor -l number_loop_repetitions -w number_of_warmup_loops -s us_to_sleep_between_multiplications -o timings_file -P scheduler_priority
```
Refer to file matilda.h for valid values of "-k mvm_kernel".
The option "-s us_to_sleep_between_multiplications" is only available if "fixed_delay_spin_loop" or "fixed_delay_stay_busy" in apps/matilda_test.cc are set to "true".

#### Example output:
  ```
    matilda_test -r 3584 -c 3456 -k 25656 -t 64 -p 64 -l 100000 -w 100000 -s 500 -o timings_microseconds.dat -P 10
    Computing 3584x3456 MVM 100000 times (+ 100000 times for warming up) with 500 micro-seconds in-between, using 64 threads starting at CPU 64.
    Using 1 56x3456 matrix block(s) per thread (756 kiB)
    26.439 +/- 0.353 us per 3584x3456 MVM. (937 GFLOP/s), min: 22.740 us (1089 GFLOP/s), max: 30.640 us (808 GFLOP/s), 3450
    MVM correct within 1.0e-05
    Writing MVM timings (32-bit floats) to timings_microseconds.dat.
```
The file timings_microseconds.dat contains the timing of each MVM execution, i.e. 200,000 32-bit floating point numbers in the example above.

#### Comparision with Intel MKL and OpenBLAS:
If installed and detected by CMake, additonal versions of the test program, `matilda_test.mkl` and `matilda_test.openblas`, will be compiled which outsource the MVM via `sgemv()`.  The number of threads and the CPU affinity must be configured externall, e.g.:
```
  OMP_NUM_THREADS=64 taskset --cpu-list 64-127 matilda_test.mkl -r 3584 -c 3456 -l 100000 -w 100000 -s 500 -o timings_microseconds.dat
```
(If using Intel MKL on AMD CPUs, refer to https://danieldk.eu/mkl-amd-zen/ for best performance.)

### Example code:
Below is an example that shows all necessary steps to use Matilda in a program.
Before the first MVM is computed, a _plan_ is created that sets up all parameters and data for the MVM and starts the worker threads.
The plan is _executed_ to compute the MVM. Between executions, the worker threads are spinning unless put to sleep by the user.

```c++
  #include "matilda.h"
  #include <x86intrin.h>
  #include <xmmintrin.h>
  #include <sys/mman.h>
  #include <sched.h>

  int main()
  {
    // Optional but recommended preparations to improve temporal determinism.
    _MM_SET_FLUSH_ZERO_MODE( _MM_FLUSH_ZERO_ON );
    _MM_SET_DENORMALS_ZERO_MODE( _MM_DENORMALS_ZERO_ON );
    mlockall( MCL_FUTURE );

    // Optional but recommended: Set the CPU affinity of the main thread before vectors are allocated and "touched" for defined data locality.
    // Vector data should be touched after allocating, e.g. via memset() to reserve pages close to a specific CPU.
    {
      cpu_set_t cps;
      CPU_ZERO( &cps );
      CPU_SET( 63, &cps ); // Choose any isolated CPU before p.first_cpu.
      sched_setaffinity( 0, sizeof(cps), &cps );
    }

    // Configure all parameters.
    Matilda::mvm_param p;
    p.matrix = ...     // Pointer to the user-provided row-major ordered, contiguous matrix used for the MVM (32-bit floating point). 
                       // Data will be copied to internal buffers.
    p.vector = ...     // Pointer to the input vector used for the MVM (32-bit floating point). Must be aligned on 64-byte boundaries.
    p.result = ...     // Pointer to the output vector of the MVM (32-bit floating point). Must be aligned on 64-byte boundaries.
    p.n_rows = 3584;   // Number of rows in the matrix and number of elements in the result vector.
    p.n_cols = 3456;   // Number of columns in the matrix and number of elements in the input vector.
    p.f16c = false;    // Store matrix internally as 16-bit floats? 
                       // Arithmetic operations are performed in 32-bit precision. 
                       // This option may or may not be useful to reduce latency by minimizing the cache space used
                       // if the loss of precision for the matrix elements can be accepted.
    p.kernel = Matilda::MvmKernel::AVX2_56; // Processing kernel used for the MVM. See matilda.h for options.
    p.n_threads = 64;  // Number of parallel threads for the multiplication. These threads will be created in addition to the calling thread.
                       // The calling thread syncronizes the threads but does is not involved with tha parallel matrix-vector multiplication.
    p.first_cpu = 64;  // Index, starting at 0, of the CPU used by the first thread. All other threads will be contiguous on subsequent CPUs.
    p.sched_policy = SCHED_FIFO; // Operating system scheduler policy for the threads. 
                                 // Note: SCHED_FIFO is a Linux real-time policy and can freeze systems if the system is not configured accordingly.
                                 // Use of SCHED_OTHER on standard Linux systems is recommended.
    p.sched_prio = 10; // Operating system schedulder priority for the threads.
                       // NOTE: high priorities with real-time policies can freeze systems if the system is not configured accordingly.
                       // Use of 0 on standard Linux systems is recommended.
    p.nice = -19;      // Operating system schedulder priority ("nice level") for the threads.


    // Create an MVM plan and start the threads.
    Matilda::mvm_plan plan( p );

    // Worker threads are spinning and consume 100% CPU time at this point.

    // Optional but recommended: Set the scheduler parameters of the main thread.
    {
      struct sched_param sched_p;
      sched_p.sched_priority = p.sched_prio;
      sched_setscheduler( 0, p.sched_policy, &sched_p );
      setpriority( PRIO_PROCESS, 0, p.nice );
    }

    // Compute the MVM as long as needed.
    while( keep_going() )
    {
      //
      // Do something, e.g.  waiting for an event and updating the vector data.
      //

      plan.execute();

      // Worker threads are spinning and consume 100% CPU time.
    }

    // If no MVMs needed for a while, threads can be put to sleep.
    plan.sleep(); 

    //
    // Do something else meanwhile
    //

    // If MVMs are needed again, sleeping threads must be woken up first.
    plan.wakeup();


    // Matilda can keep the CPUs under load with dummy MVMs between requested MVMs if not performed continuoulsy.
    // This is not needed on most CPUs, and not recommended in general and the internal spin loop 
    // is normally good enough. It also provides the lowest latency. However, some particular CPUs 
    // are known to slow down if the gap between MVMs is too long.
    // As dummy MVMs introduce additional latency and cause nonessential use of the CPU, it is advised
    // to use dummy MVMs only if it has been verified that the CPUs slow down otherwise. 
    // Using this feature precautionary is discouraged.
    while( keep_going() )
    {
      // Calculate MVM normally.
      plan.execute();

      // Keep executing dummy MVMs internally.
      plan.stay_busy();

      //
      // Do something that takes a long time.
      //

      // Stop internal dummy MVMs. Call this ahead of time to allow the threads to finish the current dummy MVM to minimize latency.
      plan.get_ready();

      //
      // Do something that takes little time.
      //
    }
    return 0;
  }
```
While not tested yet, it should be possible to use multiple plans for different MVMs in one program, each using a dedicated set of CPUs.

## Credit and Licensing
Matilda was developed at the [National Solar Observatory (NSO)](https://www.nso.edu) of the United States of America, a federally funded Research and Development Center of the U.S. National Science Foundation (NSF). NSO is managed for NSF by the Association of Universities for Research in Astronomy, Inc. (AURA). 

Matilda includes AVX2 snippets written by Konstantin (http://const.me) released under the BSD-3 clause license.

Matilda is released under the BSD-3 clause license.

Copyright 2023 National Solar Observatory / AURA, https://www.nso.edu

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
