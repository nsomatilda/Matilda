// Copyright 2023 Konstantin, http://const.me
//           2023 National Solar Observatory / AURA, https://www.nso.edu
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

#include <immintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>


// Compute product of width*16 column major matrix by vector of length `width`,
// the result is a vector of length 16
void mvm_kernel_avx2_16_a( float const * mat, float const * vec, size_t width, float* rdi )
{
  // Using 2 accumulators per row to workaround data dependency on the accumulators

  __m256 a00 = _mm256_setzero_ps();
  __m256 a01 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps();
  __m256 a11 = _mm256_setzero_ps();

  size_t const maskAlign2 = ~(size_t)1;
  float const * const vecEndAligned = vec + ( width & maskAlign2 );
  while( vec < vecEndAligned )
  {
    // Broadcast 2 elements from the vector
    __m256 const v2 = _mm256_castpd_ps( _mm256_broadcast_sd( reinterpret_cast<double const *>( vec ) ) );
    vec += 2;

    // First column of the two
    __m256 v = _mm256_moveldup_ps( v2 );
    a00 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), a01 );

    // Second column
    v = _mm256_movehdup_ps( v2 );
    a10 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), a11 );

    int const distance = 32*4;
    _mm_prefetch( mat + distance, _MM_HINT_T0 );
    _mm_prefetch( mat + distance + 16, _MM_HINT_T0 );

    mat += 32;
  }

  // Handle the possible remainder
  if( 0 != ( width & 1 ) )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    a00 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), a01 );
  }

  // Reduce 32 scalars to 16
  a00 = _mm256_add_ps( a00, a10 );
  a01 = _mm256_add_ps( a01, a11 );

  // Store the products
  _mm256_store_ps( rdi, a00 );
  _mm256_store_ps( rdi + 8, a01 );
}

void mvm_kernel_avx2_16_a_f16c( float const * mat, float const * vec, size_t width, float* rdi )
{
  // Using 2 accumulators per row to workaround data dependency on the accumulators

  __m256 a00 = _mm256_setzero_ps();
  __m256 a01 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps();
  __m256 a11 = _mm256_setzero_ps();

  size_t const maskAlign2 = ~(size_t)1;
  float const * const vecEndAligned = vec + ( width & maskAlign2 );
  while( vec < vecEndAligned )
  {
    // Broadcast 2 elements from the vector
    __m256 const v2 = _mm256_castpd_ps( _mm256_broadcast_sd( reinterpret_cast<double const *>( vec ) ) );
    vec += 2;

    // First column of the two
    __m256 v = _mm256_moveldup_ps( v2 );
    a00 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), a01 );

    // Second column
    v = _mm256_movehdup_ps( v2 );
    a10 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), a11 );

    int const distance = 16*4;
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );
    mat += 16;
  }

  // Handle the possible remainder
  if( 0 != ( width & 1 ) )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    a00 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), a01 );
  }

  // Reduce 32 scalars to 16
  a00 = _mm256_add_ps( a00, a10 );
  a01 = _mm256_add_ps( a01, a11 );

  // Store the products
  _mm256_store_ps( rdi, a00 );
  _mm256_store_ps( rdi + 8, a01 );
}

// Compute product of width*16 column major matrix by vector of length `width`,
// the result is a vector of length 16
// BTW, according to godbolt.org, gcc does better than clang for this code.
void mvm_kernel_avx2_16_b( float const * mat, float const * vec, size_t width, float* rdi )
{
  // Using 4 accumulators per row, 4*16=64 scalars in 8 AVX vectors
  __m256 a00 = _mm256_setzero_ps();
  __m256 a01 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps();
  __m256 a11 = _mm256_setzero_ps();
  __m256 a20 = _mm256_setzero_ps();
  __m256 a21 = _mm256_setzero_ps();
  __m256 a30 = _mm256_setzero_ps();
  __m256 a31 = _mm256_setzero_ps();

  constexpr size_t maskAlign4 = ~(size_t)3;
  float const * const vecEndAligned = vec + ( width & maskAlign4 );
  while( vec < vecEndAligned )
  {
    // Each iteration of this loop consumes 4 elements from the vector, and 4 columns = 64 elements from the matrix

    // Broadcast 4 elements from the vector
    __m256 const v4 = _mm256_broadcast_ps( reinterpret_cast<__m128 const *>( vec ) );
    vec += 4;

    // Column #0
    __m256 v = _mm256_permute_ps( v4, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    a00 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), a01 );

    // Column #1
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 1, 1, 1, 1 ) );
    a10 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), a11 );

    // Column #2
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    a20 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), a20 );
    a21 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 40 ), a21 );

    // Column #3
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 3, 3, 3, 3 ) );
    a30 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 48 ), a30 );
    a31 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 56 ), a31 );

    mat += 64;
  }

  // Handle the remainder
  // The branches are predictable, same outcome every time this function is called
  size_t const rem = width % 4;
  if( rem == 1 )
  {
    // Column #0
    __m256 const v = _mm256_broadcast_ss( vec );
    a00 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), a01 );
  }
  else if( rem > 1 )
  {
    // Broadcast 2 elements from the vector
    __m256 const v2 = _mm256_castpd_ps( _mm256_broadcast_sd( reinterpret_cast<double const *>( vec ) ) );

    // Column #0
    __m256 v = _mm256_moveldup_ps( v2 );
    a00 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), a01 );

    // Column #1
    v = _mm256_movehdup_ps( v2 );
    a10 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), a11 );

    if( rem > 2 )
    {
      // Column #2
      v = _mm256_broadcast_ss( vec + 2 );
      a20 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), a20 );
      a21 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 40 ), a21 );
    }
  }

  // Reduce 64 accumulators to 32
  a00 = _mm256_add_ps( a00, a20 );
  a01 = _mm256_add_ps( a01, a21 );
  a10 = _mm256_add_ps( a10, a30 );
  a11 = _mm256_add_ps( a11, a31 );

  // Reduce 32 accumulators to 16
  a00 = _mm256_add_ps( a00, a10 );
  a01 = _mm256_add_ps( a01, a11 );

  // Finally, store the products
  _mm256_store_ps( rdi, a00 );
  _mm256_store_ps( rdi + 8, a01 );
}

void mvm_kernel_avx2_16_b_f16c( float const * mat, float const * vec, size_t width, float* rdi )
{
  // Using 4 accumulators per row, 4*16=64 scalars in 8 AVX vectors
  __m256 a00 = _mm256_setzero_ps();
  __m256 a01 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps();
  __m256 a11 = _mm256_setzero_ps();
  __m256 a20 = _mm256_setzero_ps();
  __m256 a21 = _mm256_setzero_ps();
  __m256 a30 = _mm256_setzero_ps();
  __m256 a31 = _mm256_setzero_ps();

  // Compute these products
  constexpr size_t maskAlign4 = ~(size_t)3;
  float const * const vecEndAligned = vec + ( width & maskAlign4 );
  while( vec < vecEndAligned )
  {
    // Each iteration of this loop consumes 4 elements from the vector, and 4 columns = 64 elements from the matrix

    // Broadcast 4 elements from the vector
    __m256 const v4 = _mm256_broadcast_ps( reinterpret_cast<__m128 const *>( vec ) );
    vec += 4;

    // Column #0
    __m256 v = _mm256_permute_ps( v4, _MM_SHUFFLE( 0, 0, 0, 0 ) );
    a00 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), a01 );

    // Column #1
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 1, 1, 1, 1 ) );
    a10 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), a11 );

    // Column #2
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 2, 2, 2, 2 ) );
    a20 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), a20 );
    a21 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 20 ) ) ), a21 );

    // Column #3
    v = _mm256_permute_ps( v4, _MM_SHUFFLE( 3, 3, 3, 3 ) );
    a30 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 24 ) ) ), a30 );
    a31 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 28 ) ) ), a31 );

    int const distance = 16*2;
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );
    mat += 32;
  }

  // Handle the remainder
  // The branches are predictable, same outcome every time this function is called
  size_t const rem = width % 4;
  if( rem == 1 )
  {
    // Column #0
    __m256 const v = _mm256_broadcast_ss( vec );
    a00 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), a01 );
  }
  else if( rem > 1 )
  {
    // Broadcast 2 elements from the vector
    __m256 const v2 = _mm256_castpd_ps( _mm256_broadcast_sd( reinterpret_cast<double const*>( vec ) ) );

    // Column #0
    __m256 v = _mm256_moveldup_ps( v2 );
    a00 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), a00 );
    a01 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), a01 );

    // Column #1
    v = _mm256_movehdup_ps( v2 );
    a10 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), a10 );
    a11 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), a11 );

    if( rem > 2 )
    {
      // Column #2
      v = _mm256_broadcast_ss( vec + 2 );
      a20 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), a20 );
      a21 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 20 ) ) ), a21 );
    }
  }

  // Reduce 64 accumulators to 32
  a00 = _mm256_add_ps( a00, a20 );
  a01 = _mm256_add_ps( a01, a21 );
  a10 = _mm256_add_ps( a10, a30 );
  a11 = _mm256_add_ps( a11, a31 );

  // Reduce 32 accumulators to 16
  a00 = _mm256_add_ps( a00, a10 );
  a01 = _mm256_add_ps( a01, a11 );

  // Finally, store the products
  _mm256_store_ps( rdi, a00 );
  _mm256_store_ps( rdi + 8, a01 );
}


void mvm_kernel_avx2_16( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    mat += 16;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
}

void mvm_kernel_avx2_16_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    mat += 8;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
}

void mvm_kernel_avx2_24( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads // prefetching not tuned for 24
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 24;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
}

void mvm_kernel_avx2_24_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    int const distance = 1*32; // distance doesn't seem to matter here for 3456x3456 matrix, speed up is present as soon as prefetching
    _mm_prefetch( mat + distance, _MM_HINT_T0 );  // prefetch 32 elements

    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;

    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    mat += 12;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
}

void mvm_kernel_avx2_32( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), acc3 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 32;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
}

void mvm_kernel_avx2_32_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    int const distance = 1*32; // distance doesn't seem to matter here for 3456x3456 matrix, speed up is present as soon as prefetching 
    _mm_prefetch( mat + distance, _MM_HINT_T0 );  // prefetch 32 elements

    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;

    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), acc3 );
    mat += 16;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
}

void mvm_kernel_avx2_40( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), acc4 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads // Untested!
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 40;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
}

void mvm_kernel_avx2_40_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), acc4 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads // Untested!
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 20;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
}

void mvm_kernel_avx2_48( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 40 ), acc5 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads // Untested!
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 48;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
}

void mvm_kernel_avx2_48_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 20 ) ) ), acc5 );
    int const distance = 32*4; // 4 fastest for 2048x2048 with 64 threads // Untested!
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 24;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
}

void mvm_kernel_avx2_56( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 40 ), acc5 );
    acc6 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 48 ), acc6 );
    mat += 56;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
  _mm256_store_ps( rdi + 48, acc6 );
}

void mvm_kernel_avx2_56_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 20 ) ) ), acc5 );
    acc6 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 24 ) ) ), acc6 );
    mat += 28;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
  _mm256_store_ps( rdi + 48, acc6 );
}

void mvm_kernel_avx2_64( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_load_ps( mat ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 8 ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 16 ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 24 ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 32 ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 40 ), acc5 );
    acc6 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 48 ), acc6 );
    acc7 = _mm256_fmadd_ps( v, _mm256_load_ps( mat + 56 ), acc7 );
    mat += 64;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
  _mm256_store_ps( rdi + 48, acc6 );
  _mm256_store_ps( rdi + 56, acc7 );
}

void mvm_kernel_avx2_64_f16c( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();
  __m256 acc4 = _mm256_setzero_ps();
  __m256 acc5 = _mm256_setzero_ps();
  __m256 acc6 = _mm256_setzero_ps();
  __m256 acc7 = _mm256_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m256 const v = _mm256_broadcast_ss( vec );
    vec++;
    acc0 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat ) ) ), acc0 );
    acc1 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 4 ) ) ), acc1 );
    acc2 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 8 ) ) ), acc2 );
    acc3 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 12 ) ) ), acc3 );
    acc4 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 16 ) ) ), acc4 );
    acc5 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 20 ) ) ), acc5 );
    acc6 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 24 ) ) ), acc6 );
    acc7 = _mm256_fmadd_ps( v, _mm256_cvtph_ps( _mm_load_si128( reinterpret_cast<__m128i const *>( mat + 28 ) ) ), acc7 );
    mat += 32;
  }

  _mm256_store_ps( rdi, acc0 );
  _mm256_store_ps( rdi + 8, acc1 );
  _mm256_store_ps( rdi + 16, acc2 );
  _mm256_store_ps( rdi + 24, acc3 );
  _mm256_store_ps( rdi + 32, acc4 );
  _mm256_store_ps( rdi + 40, acc5 );
  _mm256_store_ps( rdi + 48, acc6 );
  _mm256_store_ps( rdi + 56, acc7 );
}




