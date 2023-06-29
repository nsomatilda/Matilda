cmake_minimum_required(VERSION 3.14)
project(
  matilda
  DESCRIPTION "A library to repeatedly multiply a constant matrix with a variable vector"
  VERSION 0.1.0.0 
  LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

include(cmake/CompilerFlags.cmake)
include(CheckCXXSourceCompiles)
include(CheckCXXSymbolExists)
include(CheckIncludeFileCXX)

option( MATILDA_USE_LOCAL_RESULT_VECTOR "Enable/disable use of local result buffers " OFF )
option( MATILDA_BENCHMARK_THREADS "Enable/disable benchmarking of threads" OFF )

if( ${CMAKE_SYSTEM_NAME} MATCHES "Linux" )
  set(MATILDA_ON_LINUX ON)
  message( STATUS "Build system: Linux" )
else()
  message( WARNING "Build system " ${CMAKE_SYSTEM_NAME} " is not supported" )
endif()


check_cxx_source_compiles( "#include <x86intrin.h>
int main() { _mm_pause(); return 0; }" HAS_X86_INTRINSICS )
if( NOT HAS_X86_INTRINSICS )
  message( STATUS "x86 intrinsics not supported by compiler or compile options" )
endif()

check_cxx_source_compiles( "#include <immintrin.h>
int main() { __m256 a,b,c,d; d = _mm256_fmadd_ps( a, b, c ); return 0; }" HAS_AVX2 )
if( NOT HAS_AVX2 )
  message( STATUS "AVX2 FMA intrinsics not supported by compiler or compile options" )
endif()

check_cxx_source_compiles( "#include <immintrin.h>
int main() { __m512 a,b,c,d; d = _mm512_fmadd_ps( a, b, c ); return 0; }" HAS_AVX512F )
if( NOT HAS_AVX512F )
  message( STATUS "AVX-512 FMA intrinsics not supported by compiler or compile options" )
endif()

check_cxx_symbol_exists( posix_memalign cstdlib HAS_POSIX_MEMALIGN )
if( NOT HAS_POSIX_MEMALIGN )
  message( FATAL_ERROR "System does not provide posix_memalign()" )
endif()

find_package(Threads REQUIRED)

set( CMAKE_REQUIRED_LIBRARIES  ${CMAKE_THREAD_LIBS_INIT} )
check_cxx_symbol_exists(pthread_setname_np pthread.h HAS_PTHREAD_SETNAME_NP)
if( NOT HAS_PTHREAD_SETNAME_NP )
  message( STATUS "System does not support pthread_setname_np()" )
endif()


check_cxx_symbol_exists( _SC_PAGESIZE unistd.h HAS_SC_PAGESIZE )
if( NOT HAS_SC_PAGESIZE )
  message( SEND_ERROR "System does not support _SC_PAGESIZE" )
endif()

check_cxx_symbol_exists( _SC_NPROCESSORS_ONLN unistd.h HAS_SC_NPROCESSORS_ONLN )
if( NOT HAS_SC_NPROCESSORS_ONLN )
  message( SEND_ERROR "System does not support _SC_NPROCESSORS_ONLN" )
endif()

check_cxx_symbol_exists(setpriority sys/resource.h HAS_SETPRIORITY)
if( NOT HAS_SETPRIORITY )
  message( STATUS "System does not support setpriority()" )
endif()

check_cxx_symbol_exists(sched_setaffinity sched.h HAS_SCHED_SETAFFINITY)
if( NOT HAS_SCHED_SETAFFINITY )
  message( STATUS "System does not support sched_setaffinity()" )
endif()

check_cxx_symbol_exists(sched_setscheduler sched.h HAS_SCHED_SETSCHEDULER)
if( NOT HAS_SCHED_SETSCHEDULER )
  message( STATUS "System does not support sched_setscheduler()" )
endif()

check_cxx_symbol_exists(__NR_perf_event_open sys/syscall.h HAS_PERF_EVENT_OPEN)
if(HAS_PERF_EVENT_OPEN)
  check_cxx_symbol_exists( mmap sys/mman.h HAS_MMAP )
  if( HAS_MMAP )
    CHECK_INCLUDE_FILE_CXX( linux/perf_event.h HAS_PERF_EVENT_H )
    if( HAS_PERF_EVENT_H )
      set(MATILDA_USE_PERF ON)
    endif()
  endif()
endif()

configure_file( "${PROJECT_SOURCE_DIR}/src/version.h.cmake.in" "${PROJECT_BINARY_DIR}/version.h" )
configure_file( "${PROJECT_SOURCE_DIR}/src/config.h.cmake.in" "${PROJECT_BINARY_DIR}/config.h" )


add_subdirectory(src)
add_subdirectory(apps)