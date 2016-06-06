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
 * mesh.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: mbetten
 */



#include "mesh.hpp"
#include "utils.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Shards_CellTopology.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cassert>

int Mesh::tuple_to_node_gid(int i, int j, int k) {
  return i*(ny+1)*(nz+1) + j*(nz+1)+k;
}
void Mesh::node_gid_to_tuple(int &i, int &j, int &k, GO gid){
  i = gid/((ny+1)*(nz+1));
  j = (gid-i*(ny+1)*(nz+1))/(nz+1);
  k = gid%(nz+1);
}

template <unsigned int N_Nodes>
void Mesh::periodic_face(Mesh::Face<N_Nodes> &face) {
  bool on_edge=true;
  for (unsigned int n=0; n<N_Nodes; ++n) {
    int i,j,k;
    node_gid_to_tuple(i,j,k,face.nodes[n]);
    if (i != nx )
      on_edge=false;
  }
  if (on_edge )
    for (unsigned int n=0; n<N_Nodes; ++n) {
      int i,j,k;
      node_gid_to_tuple(i,j,k,face.nodes[n]);
      face.nodes[n] = tuple_to_node_gid(0,j,k);
    }
}

Mesh::Mesh( string &in_name, Teuchos::RCP<Teuchos::Comm<int> > mpi_, bool periodic) :comm(mpi_), is_periodic(periodic){
  typedef Intrepid::FieldContainer<FLOAT> POINT_ARRAY;
  typedef Intrepid::FieldContainer<FLOAT> JACOBIAN_ARRAY;
  typedef Intrepid::CellTools<FLOAT>       CellTools;

  const int num_tets_per_hex = 5;

  std::ifstream in(in_name.c_str(), std::ios::in);
  in >> nx >> ny >>nz;
  FLOAT dx = 2./nx, dy = 2./ny, dz = 2./nz;
  int mapping;
  in >> mapping;

  if ( is_periodic && nx%2 == 1) {
    cerr << "Periodic mesh requires an even number of mesh points in the X direction\n";
    assert(false);
  }

  std::vector<PartRange> parts;

  // Get processor count data
  rank = comm->getRank();
  comm_size = comm->getSize();

  create_partition(parts, nx, ny, nz, comm_size);

  topo = shards::getCellTopologyData<shards::Tetrahedron<4> >();
  dimension          = 3;
  num_nodes_per_elem = 4;
  num_nodes_per_edge = 2;
  num_faces_per_elem = 4;
  num_nodes_per_face = 3;
  num_edges_per_elem = 6;

  const PartRange &my_part = parts[rank];
  LO num_hexes = (my_part.second[0]-my_part.first[0])*(my_part.second[1]-my_part.first[1])*
      (my_part.second[2]-my_part.first[2]);
  num_nodes = (my_part.second[0]-my_part.first[0]+1)*(my_part.second[1]-my_part.first[1]+1)*
      (my_part.second[2]-my_part.first[2]+1);
  num_elems = num_hexes*num_tets_per_hex;

  GO total_num_faces = 4*nx*ny*nz + 2*(nx*ny*(nz+1) + nx*(ny+1)*nz + (nx+1)*ny*nz);
  if (is_periodic) total_num_faces -= 2*ny*nz;

  if ( is_periodic && ((my_part.first[0] != 0 && my_part.second[0] == nx) ||(my_part.first[0] == 0 && my_part.second[0] != nx) ) )
    num_nodes += (my_part.second[1]-my_part.first[1]+1)*(my_part.second[2]-my_part.first[2]+1);

  // Lets fill out the elem gids
  elem_gids.reserve(num_elems);
  for (LO i=my_part.first[0]; i<my_part.second[0]; ++i)
    for (LO j=my_part.first[1]; j<my_part.second[1]; ++j)
      for (LO k=my_part.first[2]; k<my_part.second[2]; ++k)
        for (int ielem = 0; ielem < num_tets_per_hex; ++ielem)
          elem_gids.push_back(num_tets_per_hex*(i*ny*nz+j*nz + k) + ielem);

  vector<FLOAT> x(num_nodes),y(num_nodes),z(num_nodes);
  node_gids.reserve(num_nodes);
  for (LO i=my_part.first[0]; i<=my_part.second[0]; ++i)
    for (LO j=my_part.first[1]; j<=my_part.second[1]; ++j)
      for (LO k=my_part.first[2]; k<=my_part.second[2]; ++k) {
        LO node_lid= node_gids.size();
        GO node_gid = tuple_to_node_gid(i,j,k);
        node_gids.push_back(node_gid);
        x[node_lid] = -1+i*dx; y[node_lid] = -1+j*dy; z[node_lid] = -1+k*dz;
      }

  // push back the unowned periodic nodes
  if ( is_periodic && comm_size > 0 )
    for (LO j=my_part.first[1]; j<=my_part.second[1]; ++j)
      for (LO k=my_part.first[2]; k<=my_part.second[2]; ++k) {
        LO node_lid = node_gids.size();
        if (my_part.second[0] == nx && my_part.first[0] != 0){
          node_gids.push_back(tuple_to_node_gid(0,j,k));
          x[node_lid] = -1+0*dx; y[node_lid] = -1+j*dy; z[node_lid] = -1+k*dz;
        }
        if (my_part.second[0] != nx && my_part.first[0] == 0){
          node_gids.push_back(tuple_to_node_gid(nx,j,k));
          x[node_lid] = -1+nx*dx; y[node_lid] = -1+j*dy; z[node_lid] = -1+k*dz;
        }
      }

  // Make a map based on the elem gids
  element_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), elem_gids, 0, comm));
  num_global_elems = element_map->getGlobalNumElements();
  assert(static_cast<GO>(num_global_elems) == num_tets_per_hex*nx*ny*nz);

  // Create node map
  node_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), node_gids, 0, comm));
  owned_node_map = Tpetra::createOneToOne<LO, GO, NodeType>(node_map);
  num_owned_nodes = owned_node_map->getNodeNumElements();
  assert(static_cast<GO>(owned_node_map->getGlobalNumElements()) == (nx+1)*(ny+1)*(nz+1));

  nodes = FLOAT_1D_VEC_ARRAY("Mesh::nodes", num_nodes);
  for (int i=0; i<num_nodes; ++i) {
    nodes(i)[0] = x[i];nodes(i)[1] = y[i];nodes(i)[2] = z[i];
  }

  if ( is_periodic ) {
    // Create the node periodic mappings.
    node_pairs = GO_ARRAY("Mesh::node_pairs", num_nodes);
    int i,j,k;
    for (int n=0; n<num_nodes; ++n) {
      node_gid_to_tuple(i,j,k,node_map->getGlobalElement(n));
      if ( i == nx && j !=0 && j != ny && k != 0 && k != nz)
        node_pairs(n) = tuple_to_node_gid(0,j,k);
      else
        node_pairs(n) = -1;
    }
  }
 
  LO cnt=0;
  TEMP_CONNECTIVITY_ARRAY etng("Mesh::elem_to_node_gids",num_elems, num_nodes_per_elem);
  TEMP_CONNECTIVITY_ARRAY etnl("Mesh::writable_elem_to_node_lids", num_elems, num_nodes_per_elem);

  std::vector<ConstLenVector<GO, 9> > hex_to_node_gids;hex_to_node_gids.reserve(num_hexes);
  for (LO i=my_part.first[0]; i<my_part.second[0]; ++i)
    for (LO j=my_part.first[1]; j<my_part.second[1]; ++j)
      for (LO k=my_part.first[2]; k<my_part.second[2]; ++k) {
        // start with the lower left
        ConstLenVector<GO, 8> nodes;
        nodes[0] = tuple_to_node_gid(i  ,j  ,k  );
        nodes[1] = tuple_to_node_gid(i  ,j  ,k+1);
        nodes[2] = tuple_to_node_gid(i  ,j+1,k  );
        nodes[3] = tuple_to_node_gid(i  ,j+1,k+1);
        nodes[4] = tuple_to_node_gid(i+1,j  ,k  );
        nodes[5] = tuple_to_node_gid(i+1,j  ,k+1);
        nodes[6] = tuple_to_node_gid(i+1,j+1,k  );
        nodes[7] = tuple_to_node_gid(i+1,j+1,k+1);

        // Lets swap the odd elements
        if ( (i+j+k)%2 != 0 ) {
          etng(cnt,0) = nodes[3]; etng(cnt,1) = nodes[1]; etng(cnt,2) = nodes[0]; etng(cnt++,3) = nodes[5];
          etng(cnt,0) = nodes[6]; etng(cnt,1) = nodes[4]; etng(cnt,2) = nodes[5]; etng(cnt++,3) = nodes[0];
          etng(cnt,0) = nodes[6]; etng(cnt,1) = nodes[2]; etng(cnt,2) = nodes[0]; etng(cnt++,3) = nodes[3];
          etng(cnt,0) = nodes[5]; etng(cnt,1) = nodes[7]; etng(cnt,2) = nodes[6]; etng(cnt++,3) = nodes[3];
          etng(cnt,0) = nodes[5]; etng(cnt,1) = nodes[6]; etng(cnt,2) = nodes[0]; etng(cnt++,3) = nodes[3];
        } else  {
          etng(cnt,0) = nodes[2]; etng(cnt,1) = nodes[3]; etng(cnt,2) = nodes[1]; etng(cnt++,3) = nodes[7];
          etng(cnt,0) = nodes[7]; etng(cnt,1) = nodes[6]; etng(cnt,2) = nodes[4]; etng(cnt++,3) = nodes[2];
          etng(cnt,0) = nodes[2]; etng(cnt,1) = nodes[0]; etng(cnt,2) = nodes[4]; etng(cnt++,3) = nodes[1];
          etng(cnt,0) = nodes[4]; etng(cnt,1) = nodes[5]; etng(cnt,2) = nodes[7]; etng(cnt++,3) = nodes[1];
          etng(cnt,0) = nodes[2]; etng(cnt,1) = nodes[4]; etng(cnt,2) = nodes[7]; etng(cnt++,3) = nodes[1];
        }
      }
  assert(cnt == num_elems);
  for (LO i=0; i<num_elems; ++i)
    for (int n=0; n<num_nodes_per_elem; ++n) {
      etnl(i, n) = node_map->getLocalElement(etng(i,n));
      assert(etnl(i, n)>=0);
    }

  // This is how we find face and edge pairs
  multimap<Face<3>, std::pair<GO, unsigned> > face_elem_mapping;
  multimap<Face<2>, std::pair<GO, unsigned> > edge_elem_mapping;
  multimap<std::pair<GO, unsigned>, Face<2> > elem_edge_mapping;
  for (int i=0; i< num_elems; ++i) {
    if ( topo == shards::getCellTopologyData<shards::Tetrahedron<4> >()) { // for future elems
      Face<3> face;
      face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,1); face.nodes[2] = etng(i,3);
      if (is_periodic) periodic_face(face);
      face_elem_mapping.insert( pair<Face<3>, std::pair<GO,unsigned> >(face, std::pair<GO,unsigned>(i,0)) );
      face.nodes[0] = etng(i,1); face.nodes[1] = etng(i,2); face.nodes[2] = etng(i,3);
      if (is_periodic) periodic_face(face);
      face_elem_mapping.insert( pair<Face<3>, std::pair<GO,unsigned> >(face, std::pair<GO,unsigned>(i,1)) );
      face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,3); face.nodes[2] = etng(i,2);
      if (is_periodic) periodic_face(face);
      face_elem_mapping.insert( pair<Face<3>, std::pair<GO,unsigned> >(face, std::pair<GO,unsigned>(i,2)) );
      face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,2); face.nodes[2] = etng(i,1);
      if (is_periodic) periodic_face(face);
      face_elem_mapping.insert( pair<Face<3>, std::pair<GO,unsigned> >(face, std::pair<GO,unsigned>(i,3)) );

      // TODO: Add periodic support
      
      int n = 0;
      ConstLenVector<GO, 2> v;
      for(int j=0; j<num_nodes_per_elem-1; ++j)
	for(int k=j+1; k<num_nodes_per_elem; ++k) {
	  v[0] = etng(i,j); v[1] = etng(i,k);
	  Face<2> edge(v);
	  edge_elem_mapping.insert(std::pair<Face<2>, std::pair<GO,unsigned> >(edge, 
									       std::pair<GO,unsigned>(i,n++)));
	}
    } else {
      assert(false); // Only hexs now
    }
  }

  for(auto iter=edge_elem_mapping.cbegin(); iter!=edge_elem_mapping.cend(); ++iter)
    elem_edge_mapping.insert(std::pair<std::pair<GO,unsigned>, Face<2> >(iter->second, iter->first));

  // This sets up the face and edge GIDS
  LO num_faces = 0;
  for (auto i = face_elem_mapping.begin(); i != face_elem_mapping.end();) {
    size_t count = face_elem_mapping.count(i->first);
    ++num_faces;
    for (size_t ii=0; ii<count; ++ii)
      ++i;
    assert (count == 1 || count == 2);
  }

  LO num_edges = 0;
  for (auto i = edge_elem_mapping.cbegin(); i!=edge_elem_mapping.cend();) {
    size_t count = edge_elem_mapping.count(i->first);
    ++num_edges;
    for (size_t ii=0; ii<count; ++ii)
      ++i;
  }


  std::vector<LO> local_face_count(comm_size), face_counts(comm_size, 0);
  local_face_count[rank] = num_faces;
  Teuchos::reduceAll<int>(*mpi_,Teuchos::REDUCE_MAX, comm_size, &local_face_count[0], &face_counts[0]);
  num_faces = 0;
  for (int i=0;i<rank; ++i)
    num_faces += face_counts[i];

  for (auto i = face_elem_mapping.begin(); i != face_elem_mapping.end();) {
    size_t count = face_elem_mapping.count(i->first);
    face_gid_map.insert(std::pair<Face<3>, GO>(i->first, num_faces++));
    for (size_t ii=0; ii<count; ++ii)
      ++i;
  }


  std::vector<LO> local_edge_count(comm_size, 0), edge_counts(comm_size, 0);
  local_edge_count[rank] = num_edges;
  Teuchos::reduceAll<int>(*mpi_,Teuchos::REDUCE_MAX, comm_size, &local_edge_count[0], &edge_counts[0]);
  num_edges = 0;
  for (int i=0;i<rank; ++i)
    num_edges += edge_counts[i];

  for (auto i=edge_elem_mapping.cbegin(); i!=edge_elem_mapping.cend();) {
    size_t count = edge_elem_mapping.count(i->first);
    edge_gid_map.insert(std::pair<Face<2>, GO>(i->first, num_edges++));
    for (size_t ii=0; ii<count; ++ii)
      ++i;
  }


  // Now we can create the mapping between the elem and faces/edges.
  elem_to_face_gids = CONNECTIVITY_ARRAY("Mesh::elem_to_face_gids", num_elems, 4);
  elem_to_edge_gids = CONNECTIVITY_ARRAY("Mesh::elem_to_edge_gids", num_elems, num_edges_per_elem);
  elem_to_face_lids = CONNECTIVITY_ARRAY("Mesh::elem_to_face_lids", num_elems, 4);
  elem_to_edge_lids = CONNECTIVITY_ARRAY("Mesh::elem_to_edge_lids", num_elems, num_edges_per_elem);
  Kokkos::deep_copy(elem_to_face_gids, -1);
  for (int i=0; i< num_elems; ++i) {
    Face<3> face;
    face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,1); face.nodes[2] = etng(i,3);
    if (is_periodic) periodic_face(face);
    elem_to_face_gids(i,0) = face_gid_map.find(Face<3>(face))->second;
    face.nodes[0] = etng(i,1); face.nodes[1] = etng(i,2); face.nodes[2] = etng(i,3);
    if (is_periodic) periodic_face(face);
    elem_to_face_gids(i,1) = face_gid_map.find(Face<3>(face))->second;
    face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,3); face.nodes[2] = etng(i,2);
    if (is_periodic) periodic_face(face);
    elem_to_face_gids(i,2) = face_gid_map.find(Face<3>(face))->second;
    face.nodes[0] = etng(i,0); face.nodes[1] = etng(i,2); face.nodes[2] = etng(i,1);
    if (is_periodic) periodic_face(face);
    elem_to_face_gids(i,3) = face_gid_map.find(Face<3>(face))->second;

    int n = 0;
    ConstLenVector<GO, 2> v;
    for(int j=0; j<num_nodes_per_elem-1; ++j)
      for(int k=j+1; k<num_nodes_per_elem; ++k) {
	v[0] = etng(i,j); v[1] = etng(i,k);
	Face<2> edge(v);
	elem_to_edge_gids(i, n++) = edge_gid_map[edge];
      }
  }

  for (int i=0; i< num_elems; ++i) {
    for (int j=0; j<4; ++j)
      assert(elem_to_face_gids(i,j) != -1);
    for(int j=0; j<num_edges_per_elem; ++j)
      assert(elem_to_edge_gids(i,j) != -1);
  }

  // create and init face mappings.
  TEMP_CONNECTIVITY_ARRAY writable_elem_face_to_elem_gids("Mesh::elem_face_to_elem_gids",num_elems, num_faces_per_elem);
  TEMP_CONNECTIVITY_ARRAY writable_elem_face_to_proc("Mesh::elem_face_to_proc",num_elems, num_faces_per_elem);
  TEMP_CONNECTIVITY_ARRAY writable_elem_face_periodic("Mesh::elem_face_to_proc",num_elems, num_faces_per_elem);
  for (int i=0; i< num_elems; ++i)
    for (int j=0; j < num_faces_per_elem; ++j)
      writable_elem_face_to_proc(i,j) = writable_elem_face_to_elem_gids(i,j) = numeric_limits<GO>::min();


  FLOAT almost_one = 0.99999999;  // This is required for roundoff

  for (auto i = face_elem_mapping.begin(); i != face_elem_mapping.end(); ++i ) {
    bool face;
    for (int idim=0; idim<3; ++idim) {
      face = true;
      for (int inode= 0; inode < 3; ++inode)
        face &= (nodes(node_map->getLocalElement(i->first.nodes[inode]))[idim] <= -almost_one);
      if ( face) {
        if ( is_periodic && idim == 0 && (nodes(node_map->getLocalElement(i->first.nodes[0]))[idim] <= -almost_one) )
          writable_elem_face_periodic(i->second.first, i->second.second) = +1;
        else
          writable_elem_face_to_elem_gids(i->second.first, i->second.second) = -1;
      }

    }
    for (int idim=0; idim<3; ++idim) {
      face = true;
      for (int inode= 0; inode < 3; ++inode)
        face &= (nodes(node_map->getLocalElement(i->first.nodes[inode]))[idim] >= +almost_one);
      if ( face) {
        if ( is_periodic && idim == 0 && (nodes(node_map->getLocalElement(i->first.nodes[0]))[idim] >= +almost_one) )
          writable_elem_face_periodic(i->second.first, i->second.second) = -1;
        else
          writable_elem_face_to_elem_gids(i->second.first, i->second.second) = -1;
      }
    }
  }

  auto iter = face_elem_mapping.begin(), end = face_elem_mapping.end();
  for ( ; iter != end; ++iter) {
    int count = face_elem_mapping.count(iter->first);
    assert (count > 0 && count <= 2);
    if ( count == 2) {
      pair<GO, unsigned> lower = iter->second, upper = (++iter)->second;
      writable_elem_face_to_elem_gids (lower.first, lower.second) = element_map->getGlobalElement(upper.first);
      writable_elem_face_to_elem_gids (upper.first, upper.second) = element_map->getGlobalElement(lower.first);
    }
  }

  // Lets now assign nodeset nodes
  std::vector<GO> nodeset_gids;
  for (LO i=0; i<num_nodes; ++i)
    if ( (!is_periodic && (x[i] <= -almost_one || x[i] >= almost_one) )  || y[i] <= -almost_one || y[i] >= almost_one || z[i] <= -almost_one || z[i] >= almost_one  )
      nodeset_gids.push_back(node_map->getGlobalElement(i));
  boundary_node_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), nodeset_gids, 0, comm));



  // Now we need to do the parallel comm to find matching faces, all faces with a -2 are off rank
  int max_cnt;
  cnt=0;
  // Check to make sure all faces are assigned.
  for (int i=0; i< num_elems; ++i)
    for (int j=0; j < num_faces_per_elem; ++j)
      if (writable_elem_face_to_elem_gids(i,j) == numeric_limits<GO>::min())
        cnt++;
  Teuchos::reduceAll<int>(*mpi_,Teuchos::REDUCE_MAX, 1, &cnt, &max_cnt);
  vector<GO> face_nodes_bcast; face_nodes_bcast.reserve(max_cnt*4);
  std::set<Face<2> > edge_bcast;
  vector<GO> edge_nodes_bcast;
  vector<GO> edge_elem_pair, edge_elem_pair_max;
  vector<GO> face_elem_pair, face_elem_pair_max;
  for (int icomm = 0; icomm < comm_size; ++icomm) {
    face_nodes_bcast.clear();
    edge_nodes_bcast.clear();
    edge_bcast.clear();
    if ( icomm == rank) {
      // Fill in the face values
      auto iter = face_elem_mapping.begin(), end = face_elem_mapping.end();
      for ( ; iter != end; ++iter)
        if ( writable_elem_face_to_elem_gids(iter->second.first, iter->second.second) < -1) {

          for (int inode=0; inode<3; ++inode)
            face_nodes_bcast.push_back(iter->first.nodes[inode]);

	  ConstLenVector<GO, 2> v;
	  v[0] = iter->first.nodes[0]; v[1] = iter->first.nodes[1];
	  edge_bcast.insert(Face<2>(v));
	  v[0] = iter->first.nodes[0]; v[1] = iter->first.nodes[2];
	  edge_bcast.insert(Face<2>(v));
	  v[0] = iter->first.nodes[1]; v[1] = iter->first.nodes[2];
	  edge_bcast.insert(Face<2>(v));
	}
    }

    for(auto iedge=edge_bcast.cbegin(); iedge!=edge_bcast.cend(); ++iedge) {
      edge_nodes_bcast.push_back((*iedge).nodes[0]);
      edge_nodes_bcast.push_back((*iedge).nodes[1]);
    }

    int size=face_nodes_bcast.size();
    Teuchos::broadcast(*mpi_, icomm, 1, &size);
    face_nodes_bcast.resize(size);
    face_elem_pair.clear();
    face_elem_pair.resize((size/3)*3, -1);  // converting from nodes per face to elem
    face_elem_pair_max.resize((size/3)*3, -1);  // converting from nodes per face to elem
    Teuchos::broadcast(*mpi_, icomm, size, &face_nodes_bcast[0]);

    int size_edge = edge_nodes_bcast.size();
    Teuchos::broadcast(*mpi_, icomm, 1, &size_edge);
    edge_nodes_bcast.resize(size_edge);
    edge_elem_pair.clear();
    edge_elem_pair.resize((size_edge/2)*2, -1);  // converting from nodes per edge to elem
    edge_elem_pair_max.resize((size_edge/2)*2, -1);  // converting from nodes per edge to elem
    Teuchos::broadcast(*mpi_, icomm, size_edge, &edge_nodes_bcast[0]);

    if ( icomm != rank ) {
      ConstLenVector<GO,3> face_nodes;
      for (unsigned i = 0; i < face_nodes_bcast.size(); i+=3 ) {
        face_nodes[0] = face_nodes_bcast[i];face_nodes[1] = face_nodes_bcast[i+1];face_nodes[2] = face_nodes_bcast[i+2];
        Face<3> face(face_nodes);
        auto iter = face_elem_mapping.begin(), end = face_elem_mapping.end();
        for ( ; iter != end; ++iter)
          if ( !(face < iter->first) && !(iter->first < face) ) {
            face_elem_pair[(i/3)*3] = element_map->getGlobalElement(iter->second.first);
            face_elem_pair[(i/3)*3+1] =  rank;
            if ( rank < icomm )
              face_elem_pair[(i/3)*3+2] = face_gid_map.find(face)->second;
            else
              face_elem_pair[(i/3)*3+2] = -1;
            break;
          }
      }

      ConstLenVector<GO, 2> edge_nodes;
      for (unsigned i = 0; i < edge_nodes_bcast.size(); i+=2 ) {
        edge_nodes[0] = edge_nodes_bcast[i];edge_nodes[1] = edge_nodes_bcast[i+1];
        Face<2> edge(edge_nodes);
        auto iter = edge_elem_mapping.begin(), end = edge_elem_mapping.end();
        for ( ; iter != end; ++iter)
          if ( !(edge < iter->first) && !(iter->first < edge) ) {
	    edge_elem_pair[(i/2)*2] =  rank;
            if ( rank < icomm )
              edge_elem_pair[(i/2)*2+1] = edge_gid_map.find(edge)->second;
            else
              edge_elem_pair[(i/2)*2+1] = -1;
            break;
          }
      }
    }
    if ( face_elem_pair.size() > 0 )
      Teuchos::reduceAll<int>(*mpi_, Teuchos::REDUCE_MAX, face_elem_pair.size(), &face_elem_pair[0], &face_elem_pair_max[0]);

    if ( edge_elem_pair.size() > 0 ) { 
      Teuchos::reduceAll<int>(*mpi_, Teuchos::REDUCE_MAX, edge_elem_pair.size(), 
      			     &edge_elem_pair[0], &edge_elem_pair_max[0]);
    }

    if(rank==icomm) {
    for (unsigned i=0; i<face_elem_pair_max.size(); i+=3)
      assert(face_elem_pair_max[i] >= 0);

    for (unsigned i=0; i<edge_elem_pair_max.size(); i+=2)
      assert(edge_elem_pair_max[i] >= 0);
    }

    if ( icomm == rank ){ // Fill out the face and edge map data
      {
	auto iter = face_elem_mapping.begin(), end = face_elem_mapping.end();
	int indx = 0;
	for ( ; iter != end; ++iter)
	  if ( writable_elem_face_to_elem_gids(iter->second.first, iter->second.second) < -1) {
	    writable_elem_face_to_elem_gids(iter->second.first, iter->second.second) =face_elem_pair_max[indx++];
	    writable_elem_face_to_proc(iter->second.first, iter->second.second) =face_elem_pair_max[indx];
	    neighboring_procs.insert(face_elem_pair_max[indx++]);
	    GO face_gid = face_elem_pair_max[indx++];
	    if ( face_gid > -1) {
	      elem_to_face_gids(iter->second.first, iter->second.second) = face_gid;
	      face_gid_map[iter->first] = face_gid;
	    }
	  }
      }
      {
	auto iedge = edge_bcast.cbegin(), end = edge_bcast.cend();
	int indx = 0;
	for ( ; iedge != end; ++iedge) {
	  GO edge_gid = edge_elem_pair_max[indx+1];
	  GO old_gid = edge_gid_map[*iedge];
	  if ( edge_gid > -1) {
	    auto ret = edge_elem_mapping.equal_range(*iedge);
	    for(auto ielem=ret.first; ielem!=ret.second; ++ielem){
	      elem_to_edge_gids(ielem->second.first, ielem->second.second) = edge_gid;
	    }
	    edge_gid_map[*iedge] = edge_gid;

	    for(int ii=0; ii<num_elems; ++ii)
	      for(int jj=0; jj<num_edges_per_elem; ++jj)
		assert(elem_to_edge_gids(ii,jj)!=old_gid);
	  }
	  indx += 2;
	}
      }
    }
  }
    // Check to make sure all faces and edges are assigned.
  for (int i=0; i< num_elems; ++i) {
    for (int j=0; j < num_faces_per_elem; ++j)
      assert(writable_elem_face_to_elem_gids(i,j) > -2);

    for(int j=0; j<num_edges_per_elem; ++j)
      assert(elem_to_edge_gids(i,j) >= 0);
  }

  // Create the face and edge maps
  std::vector<GO> face_gids(num_elems*4);
  face_gids.clear();
  for (int i=0; i< num_elems; ++i)
    for (int j=0; j<4; ++j)
      face_gids.push_back(elem_to_face_gids(i,j));
  std::sort(face_gids.begin(), face_gids.end());
  face_gids.erase(std::unique(face_gids.begin(), face_gids.end()), face_gids.end());
  face_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), face_gids, 0, comm));
  owned_face_map = Tpetra::createOneToOne<LO, GO, NodeType>(face_map);
  assert(total_num_faces == owned_face_map->getGlobalNumElements());
  for (int i=0; i< num_elems; ++i)
    for (int j=0; j<4; ++j) {
      elem_to_face_lids(i,j) = face_map->getLocalElement(elem_to_face_gids(i,j));
      assert( elem_to_face_lids(i,j) >-1);
    }
  for (LO i=0; i<num_nodes; ++i)
    apply_mapping(nodes(i)[0], nodes(i)[1], nodes(i)[2], mapping);

  std::set<GO> edge_gid_set;
  for(auto iedge=edge_gid_map.cbegin(); iedge!=edge_gid_map.cend(); ++iedge)
    edge_gid_set.insert(iedge->second);

  std::vector<GO> edge_gids(edge_gid_set.cbegin(), edge_gid_set.cend());
  edge_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), edge_gids, 0, comm));
  owned_edge_map = Tpetra::createOneToOne<LO, GO, NodeType>(edge_map);
  for (int i=0; i< num_elems; ++i)
    for (int j=0; j<num_edges_per_elem; ++j) {
      elem_to_edge_lids(i,j) = edge_map->getLocalElement(elem_to_edge_gids(i,j));
      assert( elem_to_edge_lids(i,j) >-1);
    }


  // Fill boundary edge and face maps
  // Use the centroid to determine if on boundary
  std::vector<GO> boundary_face_gids, boundary_edge_gids;
  FLOAT xavg, yavg, zavg;
  GO n1, n2, n3;
  for(auto iface=face_gid_map.cbegin(); iface!=face_gid_map.cend(); ++iface) {
    n1 = node_map->getLocalElement(iface->first.nodes[0]);
    n2 = node_map->getLocalElement(iface->first.nodes[1]);
    n3 = node_map->getLocalElement(iface->first.nodes[2]);
    xavg = (nodes(n1)[0]+nodes(n2)[0]+nodes(n3)[0])/3;
    yavg = (nodes(n1)[1]+nodes(n2)[1]+nodes(n3)[1])/3;
    zavg = (nodes(n1)[2]+nodes(n2)[2]+nodes(n3)[2])/3;
    if((xavg<=-almost_one || xavg>=+almost_one) ||
       (yavg<=-almost_one || yavg>=+almost_one) ||
       (zavg<=-almost_one || zavg>=+almost_one))
      boundary_face_gids.push_back(iface->second);
  }
  for(auto iedge=edge_gid_map.cbegin(); iedge!=edge_gid_map.cend(); ++iedge) {
    n1 = node_map->getLocalElement(iedge->first.nodes[0]);
    n2 = node_map->getLocalElement(iedge->first.nodes[1]);
    xavg = (nodes(n1)[0]+nodes(n2)[0])/2;
    yavg = (nodes(n1)[1]+nodes(n2)[1])/2;
    zavg = (nodes(n1)[2]+nodes(n2)[2])/2;
    if((xavg<=-almost_one || xavg>=+almost_one) ||
       (yavg<=-almost_one || yavg>=+almost_one) ||
       (zavg<=-almost_one || zavg>=+almost_one))
      boundary_edge_gids.push_back(iedge->second);
  }

  boundary_face_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), 
						boundary_face_gids, 0, comm));
  boundary_edge_map = Teuchos::RCP<Map>(new Map(Teuchos::OrdinalTraits<GO>::invalid(), 
						boundary_edge_gids, 0, comm));
  owned_boundary_face_map = Tpetra::createOneToOne<LO, GO, NodeType>(boundary_face_map);
  owned_boundary_edge_map = Tpetra::createOneToOne<LO, GO, NodeType>(boundary_edge_map);

  if(!periodic) {
    const int num_boundary_faces = 2*(2*nx*ny+2*nx*nz+2*ny*nz);
    assert(owned_boundary_face_map->getGlobalNumElements()==num_boundary_faces);
    const int num_boundary_edges = (2*(5*nx*ny-nx*(ny-1)-ny*(nx-1)
				       + 5*nx*nz-nx*(nz-1)-nz*(nx-1)
				       + 5*ny*nz-ny*(nz-1)-nz*(ny-1))
				    - 4*(nx+ny+nz));
    assert(owned_boundary_edge_map->getGlobalNumElements()==num_boundary_edges);
  }
  
  // Compute the face normals
  normals = FLOAT_2D_VEC_ARRAY("Mesh::normals", num_elems, num_faces_per_elem);
  for (LO i=0; i<num_elems; ++i) {
    VECTOR v1, v2;
    // faces are (0 1 3), (1 2 3), (0 3 2), (0 2 1)
    utils::subtract(v1,nodes(etnl(i,0)), nodes(etnl(i,1)));
    utils::subtract(v2,nodes(etnl(i,3)), nodes(etnl(i,1)));
    utils::cross(normals(i,0), v2,v1);
    utils::subtract(v1,nodes(etnl(i,1)), nodes(etnl(i,2)));
    utils::subtract(v2,nodes(etnl(i,3)), nodes(etnl(i,2)));
    utils::cross(normals(i,1), v2,v1);
    utils::subtract(v1,nodes(etnl(i,0)), nodes(etnl(i,3)));
    utils::subtract(v2,nodes(etnl(i,2)), nodes(etnl(i,3)));
    utils::cross(normals(i,2), v2,v1);
    utils::subtract(v1,nodes(etnl(i,0)), nodes(etnl(i,2)));
    utils::subtract(v2,nodes(etnl(i,1)), nodes(etnl(i,2)));
    utils::cross(normals(i,3), v2,v1);
    for (LO iface=0; iface < num_faces_per_elem; ++iface)
      utils::normalize(normals(i,iface));
  }

  // Now lets get Jacobian and inv-Jacobian matries
  TEMP_JACOBIAN_ARRAY writable_jacobian("Mesh""writable_jacobian",num_elems, num_nodes_per_elem, dimension, dimension);
  TEMP_JACOBIAN_ARRAY writable_inverse_jacobian("Mesh""writable_inverse_jacobian",num_elems, num_nodes_per_elem, dimension, dimension);
  determinate_jacobian = FLOAT_1D_ARRAY("determinate_jacobian", num_elems);
  JACOBIAN_ARRAY sub_jacobian(1, num_nodes_per_elem, dimension, dimension);
  JACOBIAN_ARRAY sub_inverse_jacobian(1, num_nodes_per_elem, dimension, dimension);
  POINT_ARRAY vertices(1, num_nodes_per_elem, 3), ref_vertices(1, num_nodes_per_elem, 3);
  POINT_ARRAY sub_jac_det(1, num_nodes_per_elem);

  for (int j=0; j < num_nodes_per_elem; ++j)
    for (int idim=0; idim<dimension; ++idim)
      ref_vertices(0, j, idim) = CellTools::getReferenceVertex(topo, j)[idim];

  for (int i=0; i< num_elems; ++i) {
    for (int j=0; j < num_nodes_per_elem; ++j)
      for (int idim=0; idim<dimension; ++idim)
        vertices(0, j, idim) = nodes(etnl(i,j))[idim];
    CellTools::setJacobian(sub_jacobian, ref_vertices, vertices, topo);
    CellTools::setJacobianInv(sub_inverse_jacobian, sub_jacobian);
    CellTools::setJacobianDet(sub_jac_det, sub_jacobian);
    for (int j=0; j < num_nodes_per_elem; ++j)
      assert(sub_jac_det(0, j) > 0);
    determinate_jacobian(i) = sub_jac_det(0,0);
    for (int j=0; j < num_nodes_per_elem; ++j)
      for (int idim1=0; idim1<dimension; ++idim1)
        for (int idim2=0; idim2<dimension; ++idim2) {
          writable_jacobian(i,j,idim1,idim2) = sub_jacobian(0,j,idim1,idim2);
          writable_inverse_jacobian(i,j,idim1,idim2) = sub_inverse_jacobian(0,j,idim1,idim2);
        }
  }


  jacobian = writable_jacobian;
  inverse_jacobian = writable_inverse_jacobian;  /// Jacobian info
  elem_to_node_gids = etng;
  elem_to_node_lids = etnl;
  elem_face_to_elem_gids = writable_elem_face_to_elem_gids;
  elem_face_to_proc = writable_elem_face_to_proc;
  if ( is_periodic)
    elem_face_periodic = writable_elem_face_periodic;

  Kokkos::fence();

}


void Mesh::create_partition(std::vector<PartRange> &parts, LO NX, LO NY, LO NZ, LO nparts) {

  LO_VECTOR low(3,0), hi;
  hi[0] = NX; hi[1] = NY; hi[2] = NZ;
  parts.push_back(PartRange(low,hi));

  while (static_cast<LO>(parts.size()) < nparts) {
    // Find the largest
    size_t largest = 0;
    GO largest_size=0;
    for (size_t i=0; i<parts.size(); ++i) {
      LO nx=(parts[i].second[0]-parts[i].first[0]), ny=(parts[i].second[1]-parts[i].first[1]),
          nz=(parts[i].second[2]-parts[i].first[2]);
      GO size = nx*ny*nz;
      if ( size > largest_size){
        largest_size = size;
        largest = i;
      }
    }
    // Now lets split i in the long direction in half
    LO nx=(parts[largest].second[0]-parts[largest].first[0]), ny=(parts[largest].second[1]-parts[largest].first[1]),
        nz=(parts[largest].second[2]-parts[largest].first[2]);
    PartRange &p1= parts[largest], p2;
    p2 = p1;
    if ( nx >= ny && nx >= nz ) {
      p1.second[0] = p1.first[0]+nx/2;
      p2.first[0] = p1.second[0];
    }
    else if ( ny >= nx && ny >= nz ) {
      p1.second[1] = p1.first[1]+ny/2;
      p2.first[1] = p1.second[1];
    }
    else {
      p1.second[2] = p1.first[2]+nz/2;
      p2.first[2] = p1.second[2];
    }
    parts.push_back(p2);
  }

  GO n_tot=0;
  for (size_t i=0; i<parts.size(); ++i)
    n_tot += (parts[i].second[0]-parts[i].first[0])*(parts[i].second[1]-parts[i].first[1])*
      (parts[i].second[2]-parts[i].first[2]);
  assert(n_tot == NX*NY*NZ);
}

void Mesh::apply_mapping (FLOAT &inx, FLOAT &iny, FLOAT &inz, int mapping) {
  FLOAT outx = inx, outy = iny, outz = inz;
  switch (mapping) {
  case 0:
    return;
    break;
  case 1:
    {
      double r = (1.0-fabs(inx))*(1.0-fabs(iny));
      outx= inx*cos(r)-iny*sin(r);
      outy= inx*sin(r)+iny*cos(r);
      outz= inz;
    }
    break;
  case 2:
    outx = inx*sqrt(1 - .5*iny*iny - .5*inz*inz + iny*iny*inz*inz/3.);
    outy = iny*sqrt(1 - .5*inx*inx - .5*inz*inz + inx*inx*inz*inz/3.);
    outz = inz*sqrt(1 - .5*iny*iny - .5*inx*inx + iny*iny*inx*inx/3.);
    break;
  case 3:
    outx = inx*inx;
    if (inx < 0 ) outx = -outx;
    outy=iny;
    outz=inz;
    break;
  }


  inx=outx; iny=outy; inz=outz;
}

void Mesh::dump_data() {

  const int rank = comm->getRank();

  LO n;
  char file_name[50];
  std::ofstream mesh_out;

  // Print node information
  sprintf(file_name, "nodes-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(GO inode=0; inode!=node_gids.size(); ++inode) {
    mesh_out << node_gids[inode] << ' ';
    for(int i=0; i<3; ++i)
      mesh_out << nodes(inode)[i] << ' ';
    mesh_out << endl;
  }
  mesh_out.close();

  // Print elem information
  sprintf(file_name, "elems-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(GO ielem=0; ielem<num_elems; ++ielem) {
    mesh_out << elem_gids[ielem] << ' ';
    for(int i=0; i<num_nodes_per_elem; ++i)
      mesh_out << elem_to_node_gids(ielem, i) << ' ';
    mesh_out << endl;
  }
  mesh_out.close();

  // Print face information
  sprintf(file_name, "faces-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  // Sort faces by gid
  std::map<GO, Face<3> > gid_to_face;
  for(auto iface=face_gid_map.cbegin(); 
      iface!=face_gid_map.cend();
      ++iface) {
    gid_to_face[iface->second] = iface->first;
  }
  // save face info to file
  for(auto it=gid_to_face.cbegin();
      it!=gid_to_face.cend();
      ++it) {
    mesh_out << it->first << ' ';
    for(int i=0; i<num_nodes_per_face; ++i)
      mesh_out << it->second.nodes[i] << ' ';
    mesh_out << endl;
  }
  mesh_out.close();

  // Print edge information
  sprintf(file_name, "edges-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  // Sort edges by gid
  std::map<GO, Face<2> > gid_to_edge;
  for(auto iedge=edge_gid_map.cbegin(); 
      iedge!=edge_gid_map.cend();
      ++iedge) {
    gid_to_edge[iedge->second] = iedge->first;
  }
  // save edge info to file
  for(auto it=gid_to_edge.cbegin();
      it!=gid_to_edge.cend();
      ++it) {
    mesh_out << it->first << ' ';
    for(int i=0; i<num_nodes_per_edge; ++i)
      mesh_out << it->second.nodes[i] << ' ';
    mesh_out << endl;
  }
  mesh_out.close();

  // Print owned faces
  std::vector<GO> owned_faces;
  for(int i=0; i<=owned_face_map->getMaxGlobalIndex(); ++i)
    if(owned_face_map->getLocalElement(i)>=0)
      owned_faces.push_back(i);
  sprintf(file_name, "owned-faces-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(auto it=owned_faces.cbegin(); it!=owned_faces.cend(); ++it)
    mesh_out << *it << endl;
  mesh_out.close();

  // Print owned edges
  std::vector<GO> owned_edges;
  for(int i=0; i<=owned_edge_map->getMaxGlobalIndex(); ++i)
    if(owned_edge_map->getLocalElement(i)>=0)
      owned_edges.push_back(i);
  sprintf(file_name, "owned-edges-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(auto it=owned_edges.cbegin(); it!=owned_edges.cend(); ++it)
    mesh_out << *it << endl;
  mesh_out.close();

  // Print boundary faces
  std::vector<GO> boundary_face_gids;
  for(int i=0; i<=boundary_face_map->getMaxGlobalIndex(); ++i)
    if(boundary_face_map->getLocalElement(i)>=0)
      boundary_face_gids.push_back(i);
  sprintf(file_name, "boundary-faces-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(auto it=boundary_face_gids.cbegin(); it!=boundary_face_gids.cend(); ++it)
    mesh_out << *it << endl;
  mesh_out.close();

  // Print boundary edges
  std::vector<GO> boundary_edge_gids;
  for(int i=0; i<=boundary_edge_map->getMaxGlobalIndex(); ++i)
    if(boundary_edge_map->getLocalElement(i)>=0)
      boundary_edge_gids.push_back(i);
  sprintf(file_name, "boundary-edges-%03i.dat", rank);
  mesh_out.open(file_name, std::ios::out);
  for(auto it=boundary_edge_gids.cbegin(); it!=boundary_edge_gids.cend(); ++it)
    mesh_out << *it << endl;
  mesh_out.close();

}
