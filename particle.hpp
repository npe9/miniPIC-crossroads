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
 * particle.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: mbetten
 */

#ifndef PARTICLE_HPP_
#define PARTICLE_HPP_

#include "types.hpp"
#include "random.hpp"


/// A class that holds a single particle,
class Particle {
public:
  /// Location in physical space
  VECTOR x;
  /// Location in reference space
  VECTOR x_ref;
  /// Velocity in physical space
  VECTOR v;
  /// Velocity in reference space
  VECTOR v_ref;
  /// E field at x
  VECTOR E;
  /// B field at x
  VECTOR B;
  /// Global element id
  GO ielement;
  /// random number generator for this particle
  dRandom rand;
  /// Fraction of time left in the push [0..1]
  FLOAT time_remaining;
  /// Number of real particles in a macro particle
  FLOAT weight;
  /// Index into what type of particle we have
  int type;
  /// Flag to determine if the particle is dead
  bool is_dead;


  /// Default constuctor
  KOKKOS_INLINE_FUNCTION
  Particle() : ielement(-1), rand(0), time_remaining(0.), weight(1.0), type(0), is_dead(true) {}
};


#endif /* PARTICLE_HPP_ */
