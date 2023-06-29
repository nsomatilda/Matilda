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

#include <immintrin.h>
#include <x86intrin.h>
#include <xmmintrin.h>

void mvm_kernel_avx512_16( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m512 acc = _mm512_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m512 const v = _mm512_broadcast_f32x8( _mm256_broadcast_ss( vec ) );
    vec++;
    acc = _mm512_fmadd_ps( v, _mm512_load_ps( mat ), acc );
    int const distance = 64*2;
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 32, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 48, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 16;
  }

  _mm512_store_ps( rdi, acc );
}

void mvm_kernel_avx512_32( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m512 const v = _mm512_broadcast_f32x8( _mm256_broadcast_ss( vec ) );
    vec++;
    acc0 = _mm512_fmadd_ps( v, _mm512_load_ps( mat ), acc0 );
    acc1 = _mm512_fmadd_ps( v, _mm512_load_ps( mat + 16 ), acc1 );
    int const distance = 64*2;
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 32, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 48, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 32;
  }

  _mm512_store_ps( rdi, acc0 );
  _mm512_store_ps( rdi + 16, acc1 );
}

void mvm_kernel_avx512_64( float const * mat, float const * vec, size_t width, float * rdi )
{
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  __m512 acc2 = _mm512_setzero_ps();
  __m512 acc3 = _mm512_setzero_ps();

  float const * const vecEnd = vec + width;
  while( vec < vecEnd )
  {
    __m512 const v = _mm512_broadcast_f32x8( _mm256_broadcast_ss( vec ) );
    vec++;
    acc0 = _mm512_fmadd_ps( v, _mm512_load_ps( mat ), acc0 );
    acc1 = _mm512_fmadd_ps( v, _mm512_load_ps( mat + 16 ), acc1 );
    acc2 = _mm512_fmadd_ps( v, _mm512_load_ps( mat + 32 ), acc2 );
    acc3 = _mm512_fmadd_ps( v, _mm512_load_ps( mat + 48 ), acc3 );
    int const distance = 64*2;
    _mm_prefetch(  mat + distance, _MM_HINT_T0 );      // prefetch 16 elements
    _mm_prefetch(  mat + distance + 16, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 32, _MM_HINT_T0 ); // prefetch another 16 elements
    _mm_prefetch(  mat + distance + 48, _MM_HINT_T0 ); // prefetch another 16 elements
    mat += 64;
  }

  _mm512_store_ps( rdi, acc0 );
  _mm512_store_ps( rdi + 16, acc1 );
  _mm512_store_ps( rdi + 32, acc2 );
  _mm512_store_ps( rdi + 48, acc3 );
}


