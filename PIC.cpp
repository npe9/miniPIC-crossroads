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
 * PIC.cpp
 *
 *  Created on: Nov 14, 2014
 *      Author: mbetten
 */

#include "PIC.hpp"
#include "particle_list.hpp"
#include "particle_move.hpp"
#include "particle_type.hpp"
#include "mesh.hpp"
#include "ES.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_Utils.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_CellTools.hpp"
#include <impl/Kokkos_Timer.hpp>

#ifndef KOKKOS_HAVE_CUDA
#define NOSORT_WEIGHTING
#endif

struct WeightChargeFunctor {
  typedef DeviceSpace::scratch_memory_space shmem_space ;

  WeightChargeFunctor(DataWarehouse &data) : mesh_(data.mesh_reference()),
      ES_(data.ES_reference()), particle_types_(data.particle_type_list_reference()),
      particles_(data.particles_reference()), num_dof_points_(ES_.num_dof_points_) {

    int num_elems = mesh_.num_elems;
    unsigned int team_size = TeamPolicy::team_size_recommended(*this) ;
    Kokkos::parallel_for(TeamPolicy(num_elems, team_size), *this);
    Kokkos::fence();
  }


  unsigned team_shmem_size(int team_size) const {
    return Kokkos::View<FLOAT** , shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size, num_dof_points_);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type & dev) const {
    LO ielement = dev.league_rank();
    int team_size = dev.team_size();
    int team_rank = dev.team_rank();
    LO start = particles_.start(ielement);
    LO end = start+particles_.count_for_element(ielement);
    Kokkos::View<FLOAT**, shmem_space, Kokkos::MemoryUnmanaged> contributions(dev.team_shmem(), dev.team_size(), num_dof_points_);


    for (int i=0; i<num_dof_points_; ++i)
      contributions(team_rank, i) = 0;

    for (int i=start+team_rank; i< end; i += team_size) {
      int itype = particles_.type(i);
      FLOAT weight = particles_.weight(i);
      FLOAT tot_q = weight*particle_types_.particle_info(itype).q*(6/ mesh_.determinate_jacobian(ielement));

      for (int idof=0; idof<ES_.num_dof_points_; ++idof){
        FLOAT val = 0;
        for (int icub=0; icub<ES_.num_cubature_points_; ++icub)
          val += -tot_q*ES_.weighted_basis_values_(ielement, idof, icub);
        contributions(team_rank, idof) += val;
      }
    }
    dev.team_barrier();
    for (int i=0; i<num_dof_points_; i+= team_size) {
      int index = i + team_rank;
      if (index >= num_dof_points_)
        continue;
      double val=0;
      for (int t=0; t<team_size; ++t)
        val += contributions(t, index);
      GO row = mesh_.elem_to_node_lids(ielement, index);
      Kokkos::atomic_add(&ES_.overlap_rhs_dual_view_.d_view(row, 0), val);
    }

  }

protected:
  Mesh mesh_;
  ES ES_;
  ParticleTypeList particle_types_;
  ParticleList particles_;
  int num_dof_points_;
};

KOKKOS_INLINE_FUNCTION
void PIC::operator()(PICWeightChargeTag, const LO i) const {
  int ielement = particles_.ielement(i);
  int itype = particles_.type(i);
  FLOAT weight = particles_.weight(i);
  FLOAT tot_q = weight*particle_types_.particle_info(itype).q*(6/ mesh_.determinate_jacobian(ielement));

  for (int idof=0; idof<ES_.num_dof_points_; ++idof){
    FLOAT val = 0;
    for (int icub=0; icub<ES_.num_cubature_points_; ++icub)
      val += -tot_q*ES_.weighted_basis_values_(ielement, idof, icub);
    LO row = mesh_.elem_to_node_lids(ielement, idof);
    if ( mesh_.is_periodic && mesh_.node_pairs(row) != -1)
//FIXME 
      row = mesh_.node_map->getLocalElement(mesh_.node_pairs(row));
    Kokkos::atomic_add(&ES_.overlap_rhs_dual_view_.d_view(row, 0), val);
  }
}

void PIC::weight_charge() {
  particles_ = data_.particles_reference();
  Kokkos::deep_copy(ES_.overlap_rhs_dual_view_.d_view, 0.);
  Kokkos::fence();
  num_parts_ = particles_.num_particles();
#ifdef NOSORT_WEIGHTING
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceSpace,PICWeightChargeTag>(0,num_parts_), *this);
  Kokkos::fence();
#else
  {
    Kokkos::Impl::Timer timer; data_.particles_reference().compute_start_particles_by_element(); sort_timer_ += timer.seconds();
  }
  WeightChargeFunctor wc(data_);
#endif
  Kokkos::fence();
  /// We have to do this on the hose because maps don't work on the device
  for (LO i=0; i < static_cast<LO>(mesh_.boundary_node_map->getNodeNumElements()); ++i) {
    LO index = ES_.overlap_rhs_->getMap()->getLocalElement(mesh_.boundary_node_map()->getGlobalElement(i));
    ES_.overlap_rhs_dual_view_.h_view(index,0) = 0.0;
  }

}

KOKKOS_INLINE_FUNCTION
void PIC::operator()(PICWeightEfieldTag, const LO i) const {
  int ielement = particles_.ielement(i);
   VECTOR &xref = particles_.x_ref(i);
   VECTOR &E    = particles_.E(i);
   VECTOR &B    = particles_.B(i);
   FLOAT weights[4];
   FLOAT dof_values[4];

   mesh_.getWeights(weights, xref);

   for (int idof=0; idof<4;++idof) {
     int i = mesh_.elem_to_node_lids(ielement, idof);
     dof_values[idof] = ES_.overlap_phi_dual_view_.d_view(i, 0);
   }

   E[0] = E[1] = E[2] = 0.0;
   B[0] = B[1] = B[2] = 0.0;
   for (int idof=0; idof<4;++idof)
     for (int jdof=0; jdof<4;++jdof)
       for (int idim=0;idim<3;++idim)
         E[idim] += dof_values[idof]*weights[jdof]*grad_phi_basis_values_(ielement, idof, jdof)[idim];
}

void PIC::weight_Efield() {
  Kokkos::deep_copy(ES_.overlap_phi_dual_view_.d_view, ES_.overlap_phi_dual_view_.h_view);
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceSpace,PICWeightEfieldTag>(0,particles_.num_particles()), *this);
}

KOKKOS_INLINE_FUNCTION
void PIC::operator()(PICComputeKETag, const LO i, FLOAT &sum) const {
  int itype = particles_.type(i);
  FLOAT weight = particles_.weight(i);
  FLOAT tot_m = weight*particle_types_.particle_info(itype).m;
  FLOAT v_sqr=0;
  for (int idim=0; idim<3; ++idim)
    v_sqr += particles_.v(i)[idim]*particles_.v(i)[idim];
  sum += 0.5*tot_m*v_sqr;
}


PIC::PIC(DataWarehouse &data): data_(data), mesh_(data_.mesh_reference()),
    ES_(data.ES_reference()), particle_types_(data_.particle_type_list_reference()),
    particles_(data_.particles_reference()), num_parts_(particles_.num_particles()){

  move_timer_ = weight_charge_timer_ = sort_timer_ = solve_timer_ = weight_field_timer_ = KE_timer_ = 0.0;

  // We need to set up the grad phi
  LO num_elems = data_.mesh_reference().num_elems;
  int num_nodes_per_elem = data_.mesh_reference().num_nodes_per_elem;

  grad_phi_basis_values_ = FLOAT_3D_VEC_ARRAY("PIC::grad_phi_basis_values_", num_elems, num_nodes_per_elem, num_nodes_per_elem);

  Intrepid::Basis_HGRAD_TET_C1_FEM<FLOAT, Intrepid::FieldContainer<FLOAT> > HGradBasis;

  // Save the ref locations of the grad values
  Intrepid::FieldContainer<FLOAT> refNodes(4,3);
  for (int j=0; j < num_nodes_per_elem; ++j)
    for (int idim=0; idim<3; ++idim)
      refNodes( j, idim) = Intrepid::CellTools<FLOAT>::getReferenceVertex(data_.mesh_reference().topo, j)[idim];


  Intrepid::FieldContainer<FLOAT> Grads(num_nodes_per_elem, num_nodes_per_elem, 3);
  Intrepid::FieldContainer<FLOAT> GradsTrans(1,num_nodes_per_elem, num_nodes_per_elem, 3);

  HGradBasis.getValues(Grads, refNodes, Intrepid::OPERATOR_GRAD);

  for (LO ielem = 0; ielem < num_elems; ++ielem){
    Intrepid::FieldContainer<FLOAT> JacobInv( 1, num_nodes_per_elem, 3, 3);
    for (int inode=0; inode<num_nodes_per_elem; ++inode)
      for (int idim=0; idim<3; ++idim)
        for (int jdim=0; jdim<3; ++jdim)
          JacobInv(0, inode, idim, jdim) = data_.mesh_reference().inverse_jacobian(ielem, inode, idim, jdim);

    Intrepid::FunctionSpaceTools::HGRADtransformGRAD<FLOAT>(GradsTrans, JacobInv, Grads);
    for (int inode=0; inode<num_nodes_per_elem; ++inode)
      for (int jnode=0; jnode<num_nodes_per_elem; ++jnode)
        for (int idim=0; idim<3; ++idim)
          grad_phi_basis_values_(ielem, inode, jnode)[idim] = GradsTrans(0, inode, jnode, idim);
  }
}


void
PIC::time_step(double dt) {
  if (particle_move_.is_null())
      particle_move_ =  Teuchos::RCP<ParticleMove>(new ParticleMove(data_));

  { Kokkos::fence(); Kokkos::Impl::Timer timer; weight_charge(); weight_charge_timer_ += timer.seconds(); }
  { Kokkos::fence(); Kokkos::Impl::Timer timer;data_.ES_reference().solve(1e-6); solve_timer_ += timer.seconds(); }
  { Kokkos::fence(); Kokkos::Impl::Timer timer; weight_Efield(); weight_field_timer_ += timer.seconds(); }
  { Kokkos::fence(); Kokkos::Impl::Timer timer; particle_move_->move(dt); move_timer_ += timer.seconds(); }
  Kokkos::fence(); 
#ifdef KOKKOS_HAVE_CXX11
  FLOAT KE=0;
  { Kokkos::Impl::Timer timer;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceSpace,PICComputeKETag>(0,particles_.num_particles()), *this, KE);
    KE_timer_ += timer.seconds();
    FLOAT total_KE;
    Teuchos::reduceAll(*(mesh_.comm),Teuchos::REDUCE_SUM, 1, &KE, &total_KE);
    if (mesh_.comm->getRank() == 0 )
      std::cout << "Total KE is "<< total_KE <<std::endl;
  }

#endif
}
