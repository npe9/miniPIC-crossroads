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
 * data_wearhouse.cpp
 *
 *  Created on: Oct 3, 2014
 *      Author: mbetten
 */


#include "data_warehouse.hpp"
#include "particle_list.hpp"
#include "particle_type.hpp"
#include "mesh.hpp"
#include "ES.hpp"

DataWarehouse::DataWarehouse() : mesh_(NULL), particles_(NULL), particle_types_(NULL), ES_(NULL), kill_particles_on_wall_impact_(false) {
  particle_types_ = new ParticleTypeList();
}

void DataWarehouse::create_mesh( std::string &in_name, Teuchos::RCP<Teuchos::Comm<int> > mpi_, bool periodic) {
  if (mesh_ != NULL)
    delete mesh_;
  mesh_ = new Mesh(in_name, mpi_, periodic);
}

void DataWarehouse::dump_mesh_data() {
  mesh_->dump_data();
}

void DataWarehouse::create_particle_list(std::size_t max_capacity ) {
  if (particles_ != NULL)
    delete particles_;
  particles_ = new ParticleList(max_capacity );
  if ( mesh_ )
    particles_->set_comm(mesh_->comm);
}

void DataWarehouse::create_ES() {
  if (ES_)
    delete ES_;
  assert(mesh_);
  ES_ = new class ES(*mesh_);
}
DataWarehouse::~DataWarehouse() {
  if (mesh_) delete mesh_;
  if (particles_) delete particles_;
  if (particle_types_) delete particle_types_;
  if (ES_) delete ES_;
}



Mesh DataWarehouse::mesh() const {return *mesh_;}
ParticleList DataWarehouse::particles() const {return *particles_;}
ParticleTypeList DataWarehouse::particle_type_list() const { return *particle_types_;}
class ES DataWarehouse::ES() const {return *ES_;}

Mesh & DataWarehouse::mesh_reference() {return *mesh_;}
ParticleList & DataWarehouse::particles_reference() {return *particles_;}
ParticleTypeList & DataWarehouse::particle_type_list_reference() {return *particle_types_;}
class ES & DataWarehouse::ES_reference() {return *ES_;}
