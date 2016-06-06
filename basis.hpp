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
 * basis.hpp
 *
 *  Created on: Jul 6, 2015
 *      Author: mbetten
 */

#ifndef SRC_BASIS_HPP_
#define SRC_BASIS_HPP_


#include "mesh.hpp"
#include <Intrepid_HCURL_TET_In_FEM.hpp>
#include <Intrepid_HGRAD_TET_Cn_FEM.hpp>
#include "Intrepid_DefaultCubatureFactory.hpp"
#include <typeinfo>


typedef Intrepid::Basis_HGRAD_TET_Cn_FEM<FLOAT, Intrepid::FieldContainer<FLOAT> > ES_tet_basis;
typedef Intrepid::Basis_HCURL_TET_In_FEM<FLOAT, Intrepid::FieldContainer<FLOAT> > EM_tet_basis;


/// Basis is a class which helps in the numbering of DOFs within a basis set.  It
/// tracks how man dofs live where and can go from a edge/face/center to dof GID
/// It requires a mesh pointer to operate.
template <typename BASIS_TYPE>
class Basis : public BASIS_TYPE{

public:
  typedef Intrepid::FieldContainer<FLOAT> ArrayType;
  typedef BASIS_TYPE BasisType;

  /// Constructor, requires the order of the basis function, valid 1, 2, and 3
  Basis(int order = 1) : order_(order),num_node_(0), num_edge_(0), num_face_(0), num_center_(0),
      BASIS_TYPE(order,Intrepid::POINTTYPE_EQUISPACED), mesh_(NULL) {
    assert(order_>0 && order < 4);
    if ( typeid(BASIS_TYPE) == typeid(ES_tet_basis)) {
      if ( order_ == 1 ) {
        num_node_ = 1;
      } else if ( order_ == 2 ) {
        num_node_ = 1;
        num_edge_ = 1;
      } else if ( order_ == 3 ) {
        num_node_ = 1;
        num_edge_ = 2;
        num_face_ = 1;
      }
    } else if ( typeid(BASIS_TYPE) == typeid(EM_tet_basis)) {
      if ( order_ == 1 ) {
        num_edge_ = 1;
      } else if ( order_ == 2 ) {
        num_edge_ = 2;
        num_face_ = 2;
      } else if ( order_ == 3 ) {
        num_edge_ = 3;
        num_face_ = 6;
        num_center_ = 3;
      }
    }
  }

  /// Set a mesh pointer for maps, numbering routines
  void set_mesh(Mesh *mesh) {
    mesh_ = mesh;
    set_boundary_gids();
    set_periodic_gids();
  }

  /// Given a node LID return a range of dof gids [first, second)
  std::pair<GO, GO> node_lid_to_dof_gids( LO node_lid) {
    GO gid = mesh_->node_map->getGlobalElement(node_lid);
    return std::pair<GO, GO>(gid*num_node_, (gid+1)*num_node_);
  }

  /// Given a edge LID return a range of dof gids [first, second)
  std::pair<GO, GO> edge_lid_to_dof_gids( LO edge_lid) {
    GO gid = mesh_->edge_map->getGlobalElement(edge_lid);
    GO offset = mesh_->node_map->getGlobalNumElements()*num_node_;
    return std::pair<GO, GO>(offset+gid*num_edge_, offset+(gid+1)*num_edge_);
  }

  /// Given a face LID return a range of dof gids [first, second)
  std::pair<GO, GO> face_lid_to_dof_gids( LO face_lid) {
    GO gid = mesh_->face_map->getGlobalElement(face_lid);
    GO offset = mesh_->node_map->getGlobalNumElements()*num_node_ +
        mesh_->edge_map->getGlobalNumElements()*num_edge_;
    return std::pair<GO, GO>(offset+gid*num_face_, offset+(gid+1)*num_face_);
  }

  /// Given a element LID return a range of dof gids [first, second)
  std::pair<GO, GO> center_lid_to_dof_gids( LO elem_lid) {
    GO gid = mesh_->face_map->getGlobalElement(elem_lid);
    GO offset = mesh_->node_map->getGlobalNumElements()*num_node_ +
        mesh_->edge_map->getGlobalNumElements()*num_edge_ +
        mesh_->face_map->getGlobalNumElements()*num_face_ ;
    return std::pair<GO, GO>(offset+gid*num_center_, offset+(gid+1)*num_center_);
  }

  /// Return a stardard vector of all DOF gids in a given element
  std::vector<GO> elem_dof_gids(LO elem_lid) {
    std::vector<GO> gids(num_node_*mesh_->num_nodes_per_elem+num_edge_*mesh_->num_edges_per_elem+num_face_*mesh_->num_faces_per_elem+num_center_);
    gids.clear();
    std::pair<GO, GO> p;

    for (int i=0; i<mesh_->num_nodes_per_elem; ++i) {
      LO lid = mesh_->elem_to_node_lids(elem_lid, i);
      p = node_lid_to_dof_gids(lid);
      for (GO e=p.first; e<p.second; ++e)
        gids.push_back(e);
    }
    for (int i=0; i<mesh_->num_edges_per_elem; ++i) {
      LO lid = mesh_->elem_to_edge_lids(elem_lid, i);
      p = edge_lid_to_dof_gids(lid);
      for (GO e=p.first; e<p.second; ++e)
        gids.push_back(e);
    }
    for (int i=0; i<mesh_->num_faces_per_elem; ++i) {
      LO lid = mesh_->elem_to_face_lids(elem_lid, i);
      p = edge_lid_to_dof_gids(lid);
      for (GO f=p.first; f<p.second; ++f)
        gids.push_back(f);
    }
    p = center_lid_to_dof_gids(elem_lid);
    for (GO e=p.first; e<p.second; ++e)
      gids.push_back(e);

    return gids;
  }

  Teuchos::RCP<Intrepid::Cubature<FLOAT> > create_cubature() {
    return cub_factory_.create(mesh_->topo, order_+1);
  }

  //FIXME add this function
  /// Set the boundary gids map
  void set_boundary_gids() {
    // loop over all nodes, edges, faces,edges and add the GID if it is in mesh_->boundary_???_map
    std::vector<GO> gids;
    for (LO i=0; i<mesh_->boundary_node_map->getNodeNumElements(); ++i) {
      LO lid = mesh_->node_map->getLocalElement(mesh_->boundary_node_map->getGlobalElement(i));
      std::pair<GO, GO> g = node_lid_to_dof_gids(lid);
      for (GO j=g.first; j<g.second; ++j)
        gids.push_back(j);
    }

    for (LO i=0; i<mesh_->boundary_edge_map->getNodeNumElements(); ++i) {
      LO lid = mesh_->edge_map->getLocalElement(mesh_->boundary_edge_map->getGlobalElement(i));
      std::pair<GO, GO> g = edge_lid_to_dof_gids(lid);
      for (GO j=g.first; j<g.second; ++j)
        gids.push_back(j);
    }

    for (LO i=0; i<mesh_->boundary_face_map->getNodeNumElements(); ++i) {
      LO lid = mesh_->face_map->getLocalElement(mesh_->boundary_face_map->getGlobalElement(i));
      std::pair<GO, GO> g = face_lid_to_dof_gids(lid);
       for (GO j=g.first; j<g.second; ++j)
         gids.push_back(j);
     }

    std::sort(gids.begin(), gids.end());
    gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
    boundary_gids_ = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), gids, 0, mesh_->comm));

  }

  /// return if a DOF GID is part of a boundary element
  bool is_boundary_gid(GO gid) {
    return boundary_gids_->isNodeGlobalElement(gid);
  }

  //FIXME add this function
  /// Set the periodic gids map
  void set_periodic_gids() {
    // loop over all nodes, edges, faces,edges and add the GID if it is in mesh_->periodic_???_map
    std::vector<GO> gids;
    if (mesh_->is_periodic == false) {
      periodic_gids_ = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), gids, 0, mesh_->comm));
      return;
    }

    assert(order_ == 1 && typeid(BASIS_TYPE) == typeid(ES_tet_basis));

    for (LO i=0; i<mesh_->node_map->getNodeNumElements(); ++i) {
      if ( mesh_->node_pairs(i) == -1 )
        continue;
      std::pair<GO, GO> g = node_lid_to_dof_gids(i);
      for (GO j=g.first; j<g.second; ++j)
        gids.push_back(j);
    }
    std::sort(gids.begin(), gids.end());
    gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
    periodic_gids_ = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), gids, 0, mesh_->comm));

    for (LO i=0; i<mesh_->node_map->getNodeNumElements(); ++i) {
      if ( mesh_->node_pairs(i) == -1 )
        continue;
      std::pair<GO, GO> g = node_lid_to_dof_gids(i);
      periodic_pairs_.insert(std::pair<GO,GO>(g.first,mesh_->node_pairs(i) ));
    }


  }

  /// return if a DOF GID is part of a boundary element
  bool is_periodic_gid(GO gid) {
    if ( mesh_->is_periodic)
      return periodic_gids_->isNodeGlobalElement(gid);
    return false;
  }

  GO periodic_pair(GO gid) {
    assert(periodic_pairs_.find(gid) != periodic_pairs_.end());
    return periodic_pairs_[gid];
  }
protected:

  int order_;
  int num_node_;
  int num_edge_;
  int num_face_;
  int num_center_;

  Mesh* mesh_;

  Teuchos::RCP<Map> boundary_gids_;
  Teuchos::RCP<Map> periodic_gids_;

  Intrepid::DefaultCubatureFactory<FLOAT>  cub_factory_;
  std::map<GO, GO> periodic_pairs_;
};

typedef Basis< ES_tet_basis > ESBasis;
typedef Basis< EM_tet_basis > EMBasis;


#endif /* SRC_EM_BASIS_HPP_ */
