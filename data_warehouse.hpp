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
 * data_wearhouse.hpp
 *
 *  Created on: Oct 3, 2014
 *      Author: mbetten
 */

#ifndef DATA_WEARHOUSE_HPP_
#define DATA_WEARHOUSE_HPP_

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"
#include <string>

class ParticleList;
class Mesh;
class ParticleTypeList;
class ES;

class DataWarehouse {
public:
  DataWarehouse();
  ~DataWarehouse();

  Mesh mesh() const;
  ParticleList particles() const;
  ParticleTypeList particle_type_list() const;
  class ES ES() const;

  Mesh & mesh_reference();
  ParticleList & particles_reference();
  ParticleTypeList & particle_type_list_reference();
  class ES & ES_reference();

  /// Creates a mesh, must be called before create_ES
  void create_mesh( std::string &in_name, Teuchos::RCP<Teuchos::Comm<int> > mpi_, bool periodic=false);
  /// Save mesh data
  void dump_mesh_data();
  /// Creates a particle list of the capacity listed
  void create_particle_list(std::size_t max_capacity = 32768*1024);
  /// Creates an ES solver, used the mesh locally stored
  void create_ES();

  /// Set the flag if we kill particles on wall impact
  void set_kill_particles_on_wall_impact( bool flag) {kill_particles_on_wall_impact_=flag;}
  ///This function returns a bool if killing particles is set
  bool kill_particles_on_wall_impact() {return kill_particles_on_wall_impact_;}
private:
  // privatize copy constructors so we can't pass by value, we only want one copy
  DataWarehouse(const DataWarehouse &w):mesh_(NULL), particles_(NULL), particle_types_(NULL), ES_(NULL) {}
  DataWarehouse & operator = (const DataWarehouse &w) { return *this;}

protected:
  Mesh *mesh_;
  ParticleList *particles_;
  ParticleTypeList *particle_types_;
  class ES *ES_;
  bool kill_particles_on_wall_impact_;
};



#endif /* DATA_WEARHOUSE_HPP_ */
