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
 * particle_type.hpp
 *
 *  Created on: Nov 4, 2014
 *      Author: mbetten
 */

#ifndef SRC_PARTICLE_TYPE_HPP_
#define SRC_PARTICLE_TYPE_HPP_

#include "types.hpp"

class ParticleType {
public:
  FLOAT q;
  FLOAT m;
  FLOAT q_over_m;
  KOKKOS_INLINE_FUNCTION
  ParticleType(FLOAT q_ = -1, FLOAT m_ = 1) :
    q(q_), m(m_), q_over_m(q/m) {}
protected:

};



class ParticleTypeList {
#define MAX_NUMBER_OF_PARTICLES 10
public:
  typedef Kokkos::View<ParticleType[MAX_NUMBER_OF_PARTICLES]> ParticleTypes;
  /// Construct a simple particle type list and add
  /// two particle types, e- and H+
  /// We are using electron based units so e- q/m are both 1
  ParticleTypeList() :types_("ParticleTypeList::types") {
    names_.push_back("e-");
    names_.push_back("e+");
    names_.push_back("H+");
    types_(0) = ParticleType(-1, 1);
    types_(1) = ParticleType( 1, 1);
    types_(2) = ParticleType( 1, 1837);

    count_ =3;
  }
  void add_particle_type(const ParticleType t, const std::string &name){
    types_(count_++) = t;
    names_.push_back(name);
  }
  KOKKOS_INLINE_FUNCTION
  ParticleType& particle_info(int i) const{ return types_[i];};

protected:
  ParticleTypes types_;
  std::vector<std::string> names_;
  int count_;
};

#endif /* SRC_PARTICLE_TYPE_HPP_ */
