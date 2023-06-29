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

template <int simd_rows>
void mvm_kernel_omp_simd( const float * mat, const float * vec, size_t width, float * rdi )
{
  const float * const vecEnd = vec + width;

  float acc[simd_rows] __attribute__((aligned(64)));
  #pragma omp simd aligned(acc,mat: 64)
  for( int i = 0; i < simd_rows; i++ )
    acc[i] = mat[i] * *vec;

  mat += simd_rows;
  vec++;

  while( vec < vecEnd-1 )
  {
    #pragma omp simd aligned(acc,mat: 64)
    for( int i = 0; i < simd_rows; i++ )
      acc[i] += mat[i] * *vec;
    mat += simd_rows;
    vec++;
  }

  #pragma omp simd aligned(acc,mat,rdi: 64)
  for( int i = 0; i < simd_rows; i++ )
    rdi[i] = acc[i] + mat[i] * *vec;
}
