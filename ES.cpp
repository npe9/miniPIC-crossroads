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
 * ES.cpp
 *
 *  Created on: Oct 27, 2014
 *      Author: mbetten
 */

#include "ES.hpp"
#include "CG_solve.hpp"
#include "basis.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_Utils.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include <MatrixMarket_Tpetra.hpp>

ES::ES(Mesh &mesh) :mesh_(mesh) {

  const size_t ncols_max=100;
  // Define basis
  Intrepid::DefaultCubatureFactory<FLOAT>  cub_factory;

  ESBasis basis;
  basis.set_mesh(&mesh_);
  Teuchos::RCP<Intrepid::Cubature<FLOAT> > cubature = basis.create_cubature();
  num_cubature_points_ = cubature->getNumPoints();

  num_dof_points_ = basis.getCardinality();

  Intrepid::FieldContainer<FLOAT> GVals(num_dof_points_, num_cubature_points_);
  Intrepid::FieldContainer<FLOAT> Grads(num_dof_points_, num_cubature_points_, 3);


  cubature_points_ = FLOAT_1D_VEC_ARRAY("ES::Cubature_points", num_cubature_points_);
  weighted_basis_values_ = FLOAT_3D_ARRAY("ES::weighted_basis_values_", mesh_.num_elems, num_dof_points_, num_cubature_points_);

  Intrepid::FieldContainer<FLOAT> cubPoints(num_cubature_points_, 3);
  Intrepid::FieldContainer<FLOAT> cubWeights(num_cubature_points_);

  cubature->getCubature(cubPoints, cubWeights);

  for (int c = 0; c< num_cubature_points_; ++c) {
    for (int idim=0; idim<3; ++idim)
      cubature_points_(c)[idim]= cubPoints(c,idim);
  }



 // Evaluate basis values and gradients at cubature points
    basis.getValues(GVals, cubPoints, Intrepid::OPERATOR_VALUE);
    basis.getValues(Grads, cubPoints, Intrepid::OPERATOR_GRAD);

    Teuchos::RCP<Graph> graph(new Graph(mesh_.owned_node_map, ncols_max));
    for (LO ielem=0; ielem < mesh_.num_elems; ++ielem) {
      std::vector<GO> dofs(num_dof_points_);

      dofs = basis.elem_dof_gids(ielem);

      assert(dofs.size() == num_dof_points_);

      for (int i=0; i<num_dof_points_; ++i) {
        bool boundary_node = basis.is_boundary_gid(dofs[i]);
        bool periodic_node = basis.is_periodic_gid(dofs[i]);
        if (periodic_node)
          dofs[i] = basis.periodic_pair(dofs[i]);
        if ( boundary_node ) {
          std::vector<GO> node(1,dofs[i]);
          graph->insertGlobalIndices(dofs[i], node);
        } else {
          graph->insertGlobalIndices(dofs[i], dofs);
        }
      }
    }
    if ( mesh_.is_periodic) 
      for (LO i=0; i<mesh_.node_pairs.dimension_0(); ++i) {
        vector<GO> nodes(2, mesh_.node_map->getGlobalElement(i));
        nodes[1] = mesh_.node_pairs(i);
        if (nodes[1] > -1 )
          graph->insertGlobalIndices(nodes[0], nodes);
      }

    graph->fillComplete();

    matrix_ = Teuchos::RCP<Matrix>(new Matrix(graph));
    phi_ = Teuchos::RCP<Vector>(new Vector(mesh_.owned_node_map));
    overlap_phi_ = Teuchos::RCP<Vector>(new Vector(mesh_.node_map));
    rhs_ = Teuchos::RCP<Vector>(new Vector(mesh_.owned_node_map));
    overlap_rhs_ = Teuchos::RCP<Vector>(new Vector(mesh_.node_map));
    phi_dev_ = Teuchos::RCP<DeviceVector>(new DeviceVector(mesh_.owned_node_map_dev));
    overlap_phi_dev_ = Teuchos::RCP<DeviceVector>(new DeviceVector(mesh_.node_map_dev));
    rhs_dev_ = Teuchos::RCP<DeviceVector>(new DeviceVector(mesh_.owned_node_map_dev));
    overlap_rhs_dev_ = Teuchos::RCP<DeviceVector>(new DeviceVector(mesh_.node_map_dev));
    export_ = Teuchos::RCP<Export>(new Export(mesh_.node_map, mesh_.owned_node_map));
    import_ = Teuchos::RCP<Import>(new Import(mesh_.owned_node_map, mesh_.node_map));

    Intrepid::FieldContainer<FLOAT> nodes(1, mesh_.num_nodes_per_elem, 3);
    Intrepid::FieldContainer<FLOAT> Jacobian(1, num_cubature_points_, 3, 3);
    Intrepid::FieldContainer<FLOAT> JacobInv(1, num_cubature_points_, 3, 3);
    Intrepid::FieldContainer<FLOAT> JacobDet(1, num_cubature_points_);

    Intrepid::FieldContainer<FLOAT> localStiffMatrix(1, num_dof_points_, num_dof_points_);
    Intrepid::FieldContainer<FLOAT> localRHS(1, num_dof_points_);
    Intrepid::FieldContainer<FLOAT> weightedMeasure(1, num_cubature_points_);
    Intrepid::FieldContainer<FLOAT> interpolatedRHS(1, num_cubature_points_);
    Intrepid::FieldContainer<FLOAT> GradsTransformed(1, num_dof_points_, num_cubature_points_, 3);
    Intrepid::FieldContainer<FLOAT> GradsTransformedWeighted(1, num_dof_points_, num_cubature_points_, 3);

    // constant RHS
    for (int icub=0; icub<num_cubature_points_; ++icub)
      interpolatedRHS(0,icub) = 1.0;
    for (LO ielem=0; ielem < mesh_.num_elems; ++ielem) {
      for (int inode = 0; inode < mesh_.num_nodes_per_elem; ++inode)
        for (int idim = 0; idim < 3; ++idim)
          nodes(0,inode, idim) = mesh_.nodes(mesh_.elem_to_node_lids(ielem, inode)) [idim];
      Intrepid::CellTools<FLOAT>::setJacobian(Jacobian, cubPoints, nodes, mesh_.topo);
      Intrepid::CellTools<FLOAT>::setJacobianInv(JacobInv, Jacobian );
      Intrepid::CellTools<FLOAT>::setJacobianDet(JacobDet, Jacobian );
      // ************************** Compute element HGrad stiffness matrices *******************************

      // transform to physical coordinates
      Intrepid::FunctionSpaceTools::HGRADtransformGRAD<FLOAT>(GradsTransformed, JacobInv, Grads);

      // compute weighted measure
      Intrepid::FunctionSpaceTools::computeCellMeasure<FLOAT>(weightedMeasure, JacobDet, cubWeights);

      // multiply values with weighted measure
      Intrepid::FunctionSpaceTools::multiplyMeasure<FLOAT>(GradsTransformedWeighted,
          weightedMeasure, GradsTransformed);

      // integrate to compute element stiffness matrix
      Intrepid::FunctionSpaceTools::integrate<FLOAT>(localStiffMatrix,
          GradsTransformed, GradsTransformedWeighted, Intrepid::COMP_BLAS);
      std::vector<GO> col(1);
      std::vector<FLOAT> val(1);
      std::vector<GO> dofs = basis.elem_dof_gids(ielem);
      for (int idof=0; idof<num_dof_points_; ++idof) {
        GO row = dofs[idof], old_row = row;
        bool periodic_node = basis.is_periodic_gid(row);
        // Handle periodic nodes
        if ( periodic_node )
          row = basis.periodic_pair(row);
        for (int jdof=0; jdof<num_dof_points_; ++jdof) {
          col[0] = dofs[jdof];
          if (basis.is_periodic_gid(col[0]) )
            col[0]= basis.periodic_pair(col[0]);

          val[0] = localStiffMatrix(0, idof, jdof);
          bool boundary_node = basis.is_boundary_gid(row);
          if ( boundary_node )
            continue ;

          matrix_->sumIntoGlobalValues(row, col, val);  // Fill in regular and periodic node
          if (periodic_node) {  // Fix the ghost data
            std::vector<GO> periodic_cols(2,old_row);
            std::vector<FLOAT> periodic_vals(2, 1.0);
            periodic_cols[1] =  basis.periodic_pair(old_row);
            periodic_vals[0] = -1.0;
            matrix_->replaceGlobalValues(periodic_cols[0], periodic_cols, periodic_vals);
          }
        }
      }
      // Lets cache data for computing the RHS now
      for (int idof=0; idof<num_dof_points_; ++idof)
        for (int icub=0; icub<num_cubature_points_; ++icub)
          weighted_basis_values_(ielem, idof, icub) = weightedMeasure(0,icub)* GVals(idof, icub);
    }

    LO num_boundary_nodes = mesh_.boundary_node_map->getNodeNumElements();
    for( LO inode=0; inode < num_boundary_nodes; ++inode) {
      GO row = mesh_.boundary_node_map->getGlobalElement(inode);
      std::vector<GO> col(1, row);
      std::vector<FLOAT> val(1,1.0);
      matrix_->replaceGlobalValues(row, col, val);
    }
    matrix_->fillComplete();

    rhs_dual_view_ = rhs_dev_->getDualView();
    overlap_rhs_dual_view_ = overlap_rhs_dev_->getDualView();
    overlap_rhs_dual_view_host_ = overlap_rhs_->getDualView();
    overlap_phi_dual_view_host_ = overlap_phi_->getDualView();
    phi_dual_view_ = phi_dev_->getDualView();
    overlap_phi_dual_view_ = overlap_phi_dev_->getDualView();
}


bool ES::solve(FLOAT tol) {
  Kokkos::deep_copy(overlap_rhs_dual_view_host_.h_view, overlap_rhs_dual_view_.d_view);
  rhs_->doExport(*overlap_rhs_, *export_, Tpetra::ADD);
  bool result = BiCGStab_solve(matrix_, rhs_, phi_, tol);

  overlap_phi_->doImport(*phi_, *import_, Tpetra::INSERT);
  Kokkos::deep_copy(overlap_phi_dual_view_.d_view, overlap_phi_dual_view_host_.h_view);


  if ( false ) {
    Tpetra::MatrixMarket::Writer<Matrix>::writeSparseFile("matrix.mm", matrix_);
    Tpetra::MatrixMarket::Writer<Vector>::writeDenseFile("b.mm", rhs_);
    Tpetra::MatrixMarket::Writer<Vector>::writeDenseFile("x.mm", phi_);
  }
  return result;
}



void ES::set_analytic_RHS( FLOAT (*f)(VECTOR &x)) {
  Intrepid::FieldContainer<FLOAT> interpolatedRHS(1, num_cubature_points_);
  for (LO ielem=0; ielem < mesh_.num_elems; ++ielem) {
    for (int icub=0; icub<num_cubature_points_; ++icub) {
      VECTOR phys;
      mesh_.refToPhysical(phys, cubature_points_(icub), ielem);
      interpolatedRHS(0,icub) = f(phys);
    }
    for (int idof=0; idof<num_dof_points_; ++idof){
      FLOAT val = 0;
      for (int icub=0; icub<num_cubature_points_; ++icub)
        val += -interpolatedRHS(0,icub)*weighted_basis_values_(ielem, idof, icub);
      GO row = mesh_.elem_to_node_gids(ielem, idof);
      LO lid = mesh_.elem_to_node_lids(ielem, idof);
      if ( mesh_.boundary_node_map->isNodeGlobalElement(row))
        val = 0;
      if ( mesh_.is_periodic && mesh_.node_pairs(lid) != -1 )
        row = mesh_.node_pairs(lid);
      overlap_rhs_->sumIntoGlobalValue(row, val);
    }
  }

}
void ES::set_analytic_phi( FLOAT (*f)(VECTOR &x)) {
  Intrepid::FieldContainer<FLOAT> interpolatedRHS(1, num_cubature_points_);
  for (LO inode=0; inode < mesh_.num_nodes; ++inode) {
    VECTOR &x = mesh_.nodes(inode);
    FLOAT phi = f(x);
    GO node_gid = mesh_.node_map->getGlobalElement(inode);
    if (mesh_.owned_node_map->isNodeGlobalElement(node_gid))
      phi_->sumIntoGlobalValue(node_gid, phi);
  }
  overlap_phi_->doImport(*phi_, *export_, Tpetra::INSERT);
}
