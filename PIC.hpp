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
 * PIC.hpp
 *
 *  Created on: Nov 14, 2014
 *      Author: mbetten
 */

#ifndef SRC_PIC_HPP_
#define SRC_PIC_HPP_

#include "data_warehouse.hpp"
#include "ES.hpp"
#include "EM.hpp"
#include "particle_list.hpp"
#include "particle_type.hpp"

class ParticleMove;

/// The PIC class is the class that combines the particle
/// class with the ElectroStatic class.  This class
/// weights the charge to the mesh, solves the field and then
/// transfers the field back to the mesh
class PIC {

public:
  /// Constructor
  PIC(DataWarehouse &data);

  /// Weight the particles in the particle list to the RHS for the ES field
  void weight_charge();

  struct PICWeightChargeTag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(PICWeightChargeTag, const LO i) const;

  /// Solve the poisson equation
  void solve_for_potential();

  /// Weight the field back to the particles in the list
  void weight_Efield();

  struct PICWeightEfieldTag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(PICWeightEfieldTag, const LO i) const;

  struct PICComputeKETag {};
  KOKKOS_INLINE_FUNCTION
  void operator()(PICComputeKETag, const LO i, FLOAT &sum) const;

  /// Accelerate and move the particle
  void time_step(double dt);

  /// Output the results of the timers
  void output_timers() {
    cout << "Timers are:\nMove "<< move_timer_ <<" Weight charge "<<weight_charge_timer_ << ", Sorting "<<sort_timer_<<
        ", Solve "<< solve_timer_ << ", Weight EField "<<weight_field_timer_<< ", and KE compute "<<KE_timer_ << endl;
  }

protected:

  DataWarehouse &data_;
  Mesh mesh_;
  ES ES_;
  ParticleTypeList particle_types_;
  ParticleList particles_;
  LO num_parts_;
  FLOAT_3D_VEC_ARRAY grad_phi_basis_values_;
  FLOAT_1D_VEC_ARRAY grad_phi_;

  Teuchos::RCP<ParticleMove> particle_move_;

  double move_timer_, weight_charge_timer_, sort_timer_, solve_timer_, weight_field_timer_, KE_timer_;


};



#endif /* SRC_PIC_HPP_ */
