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
 * random.hpp
 *
 *  Created on: Sep 19, 2014
 *      Author: mbetten
 */

#ifndef RANDOM_HPP_
#define RANDOM_HPP_

/// A simple random number generator used in place of a better one.
template <typename T, T A=1103515245, T C = 12345, T M = 2147483647 >
class T_random {
public:

  KOKKOS_INLINE_FUNCTION
  T_random() : seed_(0){}

  KOKKOS_INLINE_FUNCTION
  T_random(T seed) : seed_(seed) {}

  /// Seed the random number generator
  KOKKOS_INLINE_FUNCTION
  void srand(T seed) {seed_ = seed;}

  KOKKOS_INLINE_FUNCTION
  T_random & operator = (const T_random r) { this->seed_ = r.seed_; return *this;}

  /// return a random number of length T
  KOKKOS_INLINE_FUNCTION
  T rand() {
    seed_ = (A*seed_ + C) & M;
    return seed_;
  }

  /// Return a float random number
  KOKKOS_INLINE_FUNCTION
  float frand() {
    const float f = 1./static_cast<float> (M);
    return rand()*f;
  }

  /// return a double precision random number
  KOKKOS_INLINE_FUNCTION
  double drand() {
    const double d = 1./static_cast<double> (M);
    return rand()*d;
  }
protected:
  T seed_;

};


typedef T_random<unsigned> sRandom;
typedef T_random<unsigned long, 6364136223846793005, 1442695040888963407,18446744073709551615UL > dRandom;

#endif /* RANDOM_HPP_ */
