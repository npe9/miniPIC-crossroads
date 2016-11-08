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
 * ES.hpp
 *
 *  Created on: Oct 27, 2014
 *      Author: mbetten
 */

#ifndef SRC_ES_HPP_
#define SRC_ES_HPP_

#include "mesh.hpp"
#include "types.hpp"

class PIC;
class WeightCharge;

class ES {
public:

  /// Constructor for a field solve,
  /// @param mesh Mesh data structure
  ES(Mesh &mesh);

  /// Return the value of the solution at the DOF location lid
  /// @param lid local id of the DOF, has to be in the owned map
  /// @return value of the potential field
  FLOAT phi(LO lid) { return phi_->getData()[lid]; }

  /// Return the value of the rhs at the DOF location lid
  /// @param lid local id of the DOF, has to be in the owned map
  /// @return value of the rhs
  FLOAT rhs(LO lid) { return rhs_->getData()[lid]; }

  /// Set an analytic RHS, this is used for testing purposes only
  /// @param f pointer to a function which takes in a locaiton and return a source value at that location
  void set_analytic_RHS( FLOAT (*f)(VECTOR &x));

  /// Set an analytic RHS, this is used for testing purposes only
  /// @param f pointer to a function which takes in a locaiton and return a source value at that location
  void set_analytic_phi( FLOAT (*f)(VECTOR &x));


  /// Solve the ES field given a tolerance,
  /// @param tol tolerance of the solution |error| / |rhs| < tol
  bool solve(FLOAT tol = 1.e-5);


protected:
  Mesh mesh_;
  const static int cubature_degree_ = 2;
  int num_cubature_points_;
  int num_dof_points_;
  FLOAT_1D_VEC_ARRAY cubature_points_;

  FLOAT_1D_VEC_ARRAY basis_grad_values_;
  FLOAT_1D_ARRAY basis_values_;
  FLOAT_3D_ARRAY weighted_basis_values_;

  Teuchos::RCP<Matrix> matrix_;
  Teuchos::RCP<Vector> rhs_;
  Teuchos::RCP<DeviceVector> rhs_dev_;
  VectorDualVeiw rhs_dual_view_;

  Teuchos::RCP<Vector> overlap_rhs_;
  Teuchos::RCP<DeviceVector> overlap_rhs_dev_;
  VectorDualVeiw overlap_rhs_dual_view_;
  HostVectorDualView overlap_rhs_dual_view_host_;

  Teuchos::RCP<Vector> phi_;
  Teuchos::RCP<DeviceVector> phi_dev_;
  VectorDualVeiw phi_dual_view_;

  Teuchos::RCP<Vector> overlap_phi_;
  Teuchos::RCP<DeviceVector> overlap_phi_dev_;
  VectorDualVeiw overlap_phi_dual_view_;
  HostVectorDualView overlap_phi_dual_view_host_;

  Teuchos::RCP<Export> export_;
  Teuchos::RCP<Import> import_;

  friend class PIC;
  friend class WeightChargeFunctor;
  friend class WeightEField;

};



#endif /* SRC_ES_HPP_ */
