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
#ifndef _MESH_HPP_
#define _MESH_HPP_

#include "types.hpp"
#include "const_len_vector.h"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"


#include <algorithm>
#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <map>
#include <utility>

using namespace std;
/**  Mesh class, all members are public, This class creates a pamgen mesh and
 * provides some simple routines for moving in and out of reference space.
 *
 *
 */

class Mesh {
public:

/**
 * A Mesh::Face is a class used for matching faces in elements
 */
  template <unsigned int N_Nodes>
  class Face {
  public:
    Face() {}
    Face(ConstLenVector<GO, N_Nodes> &nodes_) :nodes(nodes_) {
      std::sort(&nodes[0], &nodes[0]+N_Nodes);
    }
    Face(const Face<N_Nodes> &f) {
      for (unsigned i=0; i<N_Nodes; ++i)
        nodes[i] = f.nodes[i];
      std::sort(&nodes[0], &nodes[0]+N_Nodes);
    }
    /// COmpare function, needed for multimap
    bool operator < (const Face<N_Nodes> &face) const {
      for (unsigned i=0; i<N_Nodes; ++i)
        if (nodes[i] < face.nodes[i])
          return true;
        else if (nodes[i] > face.nodes[i])
          return false;
      return false;
    }

  public:
    ConstLenVector<GO, N_Nodes> nodes;
  };

  /// Communication info
  const Teuchos::RCP< const Teuchos::Comm< int > > comm;
  int rank;
  int comm_size;

  shards::CellTopology topo;

  int dim;
  LO num_elems;
  GO num_global_elems;
  LO num_nodes;
  LO num_owned_nodes;
  char name[100];
  int num_blocks;
  int num_node_sets;
  int num_side_sets;
  bool is_periodic;

  GO_STD_VECTOR elem_gids;
  GO_STD_VECTOR node_gids;
  FLOAT_1D_VEC_ARRAY nodes;
  FLOAT_2D_VEC_ARRAY normals;
  std::map<Face<3>, GO> face_gid_map;
  std::map<Face<2>, GO> edge_gid_map;

  /// Element data
  int num_nodes_per_elem;
  int num_nodes_per_edge;
  int num_faces_per_elem;
  int num_nodes_per_face;
  int num_edges_per_elem;
  int dimension;

  Teuchos::RCP<Map> element_map, node_map;
  Teuchos::RCP<Map> boundary_node_map, boundary_edge_map, boundary_face_map;
  Teuchos::RCP<Map> face_map;
  Teuchos::RCP<Map> edge_map;
  Teuchos::RCP<const Map> owned_node_map, owned_face_map, owned_edge_map;
  Teuchos::RCP<const Map> owned_boundary_face_map, owned_boundary_edge_map;

  // connectivity data
  CONNECTIVITY_ARRAY elem_to_node_gids;
  CONNECTIVITY_ARRAY elem_to_node_lids;

  CONNECTIVITY_ARRAY elem_to_edge_gids;
  CONNECTIVITY_ARRAY elem_to_edge_lids;

  CONNECTIVITY_ARRAY elem_to_face_gids;
  CONNECTIVITY_ARRAY elem_to_face_lids;

  CONNECTIVITY_ARRAY elem_face_to_elem_gids;
  CONNECTIVITY_ARRAY elem_face_to_proc;

  ///Periodic stuff
  CONNECTIVITY_ARRAY elem_face_periodic;
  GO_ARRAY node_pairs;

  /// Jacobian info
  JACOBIAN_ARRAY jacobian;
  JACOBIAN_ARRAY inverse_jacobian;
  FLOAT_1D_ARRAY determinate_jacobian;

  std::set<LO> neighboring_procs;

  /// Constrctor,
  /// @param in_name file name of pamgen config file
  /// @param mpi_ the Teuchos MPI communicator
  Mesh( string &in_name, Teuchos::RCP<Teuchos::Comm<int> > mpi_, bool periodic=false);

  /// Print mesh data
  void dump_data();
 
  /// Returns the nodal weights for a point in reference space
  /// @param weights output nodal weights
  /// @param ref input reference location
  KOKKOS_INLINE_FUNCTION
  void getWeights(FLOAT *weights, const ConstLenVector<FLOAT, 3> &ref) const {
    weights[0] = 1-ref[0]-ref[1]-ref[2];
    weights[1] = ref[0];
    weights[2] = ref[1];
    weights[3] = ref[2];

  }

/// Maps a reference location to a physical location
  /// @param phys output physical location
  /// @param ref input reference location
  /// @param element input element ref point is in
  KOKKOS_INLINE_FUNCTION
  void refToPhysical(ConstLenVector<FLOAT, 3> &phys, const ConstLenVector<FLOAT, 3> &ref, LO element) const {
    ConstLenVector<FLOAT, 4> weights;
    getWeights(weights, ref);
    phys[0] = phys[1] = phys[2] = 0.0;
    for (unsigned i=0; i<4; ++i)
      for (unsigned idim=0; idim < 3; ++idim)
        phys[idim] += weights[i]*nodes(elem_to_node_lids(element,i))[idim];
  }
  /// Maps a physical location to a reference location, this is iterative
  /// and can be quite expensive based on how distorted the cell is.
  /// @param ref output reference location
  /// @param phys input physical location
  /// @param element input element ref point is in
  KOKKOS_INLINE_FUNCTION
  void physToReference(ConstLenVector<FLOAT, 3> &ref, const ConstLenVector<FLOAT, 3> &phys, LO element) const {
    ConstLenVector<FLOAT, 4> weights;
    VECTOR approx_phys, delta_vec, delta_ref_vec;
    FLOAT delta=0;
    ref[0] = ref[1] = ref[2] = 0.0;
    approx_phys = nodes(elem_to_node_lids(element,0));
    delta = 0;
    for (unsigned idim=0; idim < 3; ++idim){
      delta_vec[idim] = phys[idim] - approx_phys[idim];
      delta += delta_vec[idim]*delta_vec[idim];
    }
    physToReferenceVector(delta_ref_vec, delta_vec, ref, element);
    for (unsigned idim=0; idim < 3; ++idim)
      ref[idim] += delta_ref_vec[idim];
  }
  /// This function returns a bool is a physical coordinate is in an element
  /// @return bool, is the physical vector in element
  /// @param phys_vec input physical vector
  /// @param element input element ref point is in
  KOKKOS_INLINE_FUNCTION
  bool isPhysicalInElement(const ConstLenVector<FLOAT, 3> &phys, LO element) const {
    VECTOR ref;
    physToReference(ref, phys, element);
    bool is_in = true;
    for (unsigned idim=0; idim < 3; ++idim)
      is_in &= (ref[idim]>=.0);
    is_in &= ref[0]+ref[1]+ref[2] <=1;
    return is_in;
  }
  /// Maps a vector in reference space to a physical space
  /// @param phys_vec output physical vector
  /// @param ref_vec input reference vector
  /// @param ref input reference location
  /// @param element input element ref point is in
  KOKKOS_INLINE_FUNCTION
  void refToPhysicalVector(ConstLenVector<FLOAT, 3> &phys_vec, const ConstLenVector<FLOAT, 3> &ref_vec, const ConstLenVector<FLOAT, 3> &ref, LO element) const  {
    phys_vec[0] = phys_vec[1] = phys_vec[2] = 0.0;
      for (unsigned idim=0; idim < 3; ++idim)
        for (unsigned jdim=0; jdim < 3; ++jdim)
          phys_vec[idim] += jacobian(element, 0, idim, jdim)*ref_vec[jdim];
  }

  /// Maps a vector in physical space to a reference space
  /// @param ref_vec output reference vector
  /// @param phys_vec input physical vector
  /// @param ref input reference location
  /// @param element input element ref point is in
  KOKKOS_INLINE_FUNCTION
  void physToReferenceVector(ConstLenVector<FLOAT, 3> &ref_vec, const ConstLenVector<FLOAT, 3> &phys_vec, const ConstLenVector<FLOAT, 3> &ref, LO element) const {
    ref_vec[0] = ref_vec[1] = ref_vec[2] = 0.0;
    for (unsigned idim=0; idim < 3; ++idim)
      for (unsigned jdim=0; jdim < 3; ++jdim)
        ref_vec[idim] += inverse_jacobian(element, 0, idim, jdim)*phys_vec[jdim];
  }

protected:

  GO nx, ny, nz;

  typedef std::pair<LO_VECTOR, LO_VECTOR> PartRange;
  void create_partition(std::vector<PartRange > &parts, LO nx, LO ny, LO nz, LO nparts);

  void apply_mapping (FLOAT &x, FLOAT &y, FLOAT &z, int mapping=0);

  int tuple_to_node_gid(int i, int j, int k);
  void node_gid_to_tuple(int &i, int &j, int &k, GO gid);
  template <unsigned int N_Nodes>
  void periodic_face(Face<N_Nodes> &face);


};

#endif
