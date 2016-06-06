/**********************************************************************************

Mini-PIC 

Copyright (c) 2015, Sandia National Laboratories
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

For questions, comments or contributions contact 
Matt Bettencourt, mbetten@sandia.gov

*******************************************************************************/
/*
 * utils.hpp
 *
 *  Created on: Sep 25, 2014
 *      Author: mbetten
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cmath>

namespace utils {

/// result = a x b;
KOKKOS_INLINE_FUNCTION
void cross(VECTOR &result, const VECTOR &a, const VECTOR &b) {
  result[0] = a[1]*b[2] - a[2]*b[1];
  result[1] = a[2]*b[0] - a[0]*b[2];
  result[2] = a[0]*b[1] - a[1]*b[0];
}
/// result += a x b;
KOKKOS_INLINE_FUNCTION
void add_cross(VECTOR &result, const VECTOR &a, const VECTOR &b) {
  result[0] += a[1]*b[2] - a[2]*b[1];
  result[1] += a[2]*b[0] - a[0]*b[2];
  result[2] += a[0]*b[1] - a[1]*b[0];
}

/// return a . b;
KOKKOS_INLINE_FUNCTION
FLOAT dot(const VECTOR &a, const VECTOR &b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/// return the |x|
KOKKOS_INLINE_FUNCTION
FLOAT norm(const VECTOR &a) {
  return sqrt(dot(a,a));
}

/// normalize the vector a
KOKKOS_INLINE_FUNCTION
void normalize(VECTOR &a) {
  FLOAT factor = 1./norm(a);
  a[0] *= factor;
  a[1] *= factor;
  a[2] *= factor;
}

/// return result=a-b
KOKKOS_INLINE_FUNCTION
void subtract(VECTOR &result, const VECTOR &a, const VECTOR &b) {
  result[0] = a[0]-b[0];
  result[1] = a[1]-b[1];
  result[2] = a[2]-b[2];

}

template <typename T>
KOKKOS_INLINE_FUNCTION
T abs(const T a) {
  if (a<0) return -a;
  return a;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
T min(const T a, const T b) {
  if (a<b) return a;
  return b;
}


template <typename T>
KOKKOS_INLINE_FUNCTION
T max(const T a, const T b) {
  if (!(a<b)) return a;
  return b;
}

}
#endif /* UTILS_HPP_ */
