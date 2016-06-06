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
 * particle_list_functors.hpp
 *
 *  Created on: Oct 13, 2014
 *      Author: mbetten
 */

#ifndef PARTICLE_LIST_FUNCTORS_HPP_
#define PARTICLE_LIST_FUNCTORS_HPP_

#include "mesh.hpp"
#include "particle_list.hpp"
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_RCP.hpp>

/// This is the class that will run through the list and delete the particles, it provides the functor
class DeleteParticles: public BaseKokkosFunctor {
public:
  class CountParticlesBeyondDelete {
  public:
    typedef std::size_t value_type;
    CountParticlesBeyondDelete(ParticleList particles, int nparts, int ndelete):
      particles_(particles), nparts_(nparts), ndelete_(ndelete), nstart_(nparts-ndelete){}
    KOKKOS_INLINE_FUNCTION
    void operator() (std::size_t i, std::size_t &lsum) const {
       if (particles_.is_dead(nstart_+i) )
         lsum++;
     }
    ParticleList particles_;
    GO nparts_, ndelete_, nstart_;
  };
  DeleteParticles(ParticleList particles) :
    BaseKokkosFunctor("ParticleList::DeleteParticles"),
    particles_(particles), num_particles_(0), num_deleted_(0),
    fill_from_("ParticleList::DeleteParticles::grab_bag"), start_from_(0) {}

  /// delete the particles on the deleted_list in the particle list
  void execute();

  /// Kokkos functor to delete particle
  void operator() (std::size_t i) const;

protected:
  friend class ParticleList;
  ParticleList particles_;
  std::size_t num_particles_, num_deleted_;
  Kokkos::View<LO> fill_from_;
  std::size_t start_from_;
};


/// This class takes a list of particles in a migrate array and a list of procesors and then counts how many particles
/// goes to each processor
class CountParticleByProcessor : public BaseKokkosFunctor {
public:
  typedef LO value_type[];

  CountParticleByProcessor(Kokkos::View<LO *> proc_reference_list, Kokkos::View<Kokkos::pair<GO, int>*> migrate_list) :
    BaseKokkosFunctor("CountParticleByProcessor"), proc_reference_list_(proc_reference_list), migrate_list_(migrate_list) {
    value_count = proc_reference_list_.dimension_0();
    errors_.push_back(std::string("Invalid processor"));

  }
  /// Sum into the update array and also set the migrate_list_(i).second to the array we are summing
  /// into, i.e, the location in proc list
  KOKKOS_INLINE_FUNCTION
  void operator()( std::size_t indx , LO update[] ) const {
    LO iproc = migrate_list_(indx).second;
    for (int i=0; i<value_count; ++i)
      if ( iproc == proc_reference_list_(i) ){
        migrate_list_(indx).second=i;
        update[i]++;
        return;
      }
    push_error(0);
  }
  /// Init to zero
  KOKKOS_INLINE_FUNCTION
  void init( LO update[] ) const {
    for (int i=0; i<value_count; ++i)
      update[i] = 0;
  }
  /// Join across threads
  KOKKOS_INLINE_FUNCTION
  void join( volatile LO update[] ,
      volatile const LO input[] ) const {
    for (int i=0; i<value_count; ++i)
      update[i] += input[i];
  }
  /// The length of the vector to be reduced to
  LO value_count;
protected:
  Kokkos::View<LO *> proc_reference_list_;
  Kokkos::View<Kokkos::pair<GO, int>*> migrate_list_;
};

/// This class packs the particles in the list into arrays for migration
class PackParticlesForMigration : public BaseKokkosFunctor {
public:
  PackParticlesForMigration(ParticleList particles, Kokkos::View<Kokkos::pair<GO, int>*> migrate_list,
      Kokkos::View<Particle*> migrate_particles, Kokkos::View<LO*> index) :
    BaseKokkosFunctor("PackParticlesForMigration"),
    particles_(particles),
    migrate_list_(migrate_list),
    migrate_particles_(migrate_particles),
    index_(index){
  }
  KOKKOS_INLINE_FUNCTION
  void operator()( std::size_t indx ) const {
    Kokkos::pair<GO, int> iter = migrate_list_(indx);
    LO location = Kokkos::atomic_fetch_add(&index_(iter.second), 1);
    migrate_particles_(location) = particles_.particle(iter.first);
  }

  Kokkos::View<LO*> get_index() const {return index_;}

protected:
  ParticleList particles_;
  Kokkos::View<Kokkos::pair<GO, int>*> migrate_list_;
  Kokkos::View<Particle*>  migrate_particles_;
  Kokkos::View<LO*> index_;

};


/// This class packs the particles in the list into arrays for migration
class UnpackParticlesFromMigration : public BaseKokkosFunctor {
public:
  typedef DeviceSpace::scratch_memory_space shmem_space ;
  UnpackParticlesFromMigration(ParticleList particles,Kokkos::View<Particle*> recved_particles, Teuchos::RCP<Map> map) :
    BaseKokkosFunctor("UnpackParticlesForMigration"), particles_(particles), recved_particles_(recved_particles), element_map_(map) {}

  /// This function says how much shared memory to allocate
  /// needed for the functor
  unsigned team_shmem_size(int team_size) const {
    return  Kokkos::View<long*, shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size) ;
  }
  /// Fill functor
  KOKKOS_INLINE_FUNCTION
  void operator () (const member_type & dev) const {
    const LO league_rank = dev.league_rank();
    const LO team_size = dev.team_size();
    const LO team_rank = dev.team_rank();
    LO indx = league_rank*team_size+team_rank;
    ConstLenVector<FLOAT, 3> ref, phys;
    Kokkos::View<long*, shmem_space, Kokkos::MemoryUnmanaged> is_filled(dev.team_shmem(), team_size);

    is_filled(team_rank) = indx < static_cast<LO>(recved_particles_.dimension_0());

    if (is_filled(team_rank) )
      recved_particles_(indx).is_dead = false;
    particles_.push_back(dev, &recved_particles_(league_rank*team_size), &is_filled(0), team_size);
  }


protected:
  ParticleList particles_;
  Kokkos::View<Particle*>  recved_particles_;
  Teuchos::RCP<Map> element_map_;
};

/// This class packs the particles into the arrays owned by the particle list class for migration and actually calls the migration
/// and then unpacks them in
class MigrateParticles : public BaseKokkosFunctor {
public:
  MigrateParticles(ParticleList particles, Mesh mesh) :BaseKokkosFunctor("MigrateParticles"),
  particles_(particles), mesh_(mesh), n_remote_procs_(-1), index_("MigrateParticles::index", 0) {}
  /// Actually migrate the particles.
  void execute() {
    n_remote_procs_ = particles_.proc_reference_list_.dimension_0();

    // Count by proc
    std::vector<LO> count(n_remote_procs_), start(n_remote_procs_+1);
    CountParticleByProcessor counter(particles_.proc_reference_list_, particles_.migrate_list_);
    if ( particles_.migrate_count_() == 0 )
      for (size_t i=0; i<n_remote_procs_; ++i)
	count[i] = 0;
    else
      Kokkos::parallel_reduce(particles_.migrate_count_(), counter, &count[0]);
    Kokkos::fence();
    if (particles_.migrate_particles_.dimension_0() < particles_.migrate_count_())
      particles_.migrate_particles_ = Kokkos::View<Particle*>("MigrateParticles::MigrateParticles", particles_.migrate_count_());

    // Compute starting locations in migrate list
    if ( static_cast<LO>(index_.dimension_0()) <= n_remote_procs_)
      index_ = Kokkos::View<int*>("MigrateParticles::index", n_remote_procs_+1);
    index_(0) = 0;
    for (int n=1; n<n_remote_procs_+1; ++n) {
      index_(n) = index_(n-1)+count[n-1];
      start[n] = index_(n);
    }

    // Pack up the particles into a single list
    PackParticlesForMigration packer(particles_, particles_.migrate_list_,
        particles_.migrate_particles_, index_);
    Kokkos::parallel_for(particles_.migrate_count_(), packer);
    Kokkos::fence();

    // Communicate the sizes
    std::vector<Teuchos::RCP<Teuchos::CommRequest<int> > > send_reqs(n_remote_procs_), recv_reqs(n_remote_procs_);
    std::vector<int> recv_count(n_remote_procs_, 0);
    for (int n=0; n<n_remote_procs_; ++n) {
      Teuchos::ArrayView<char> rc((char*)&recv_count[n], sizeof(int));
      recv_reqs[n] = particles_.comm_->ireceive(rc, particles_.proc_reference_list_(n));
      Teuchos::ArrayView<char> sc((char*)&count[n], sizeof(int));
      send_reqs[n] = particles_.comm_->isend(sc, particles_.proc_reference_list_(n));
    }
    particles_.comm_->waitAll(send_reqs);
    particles_.comm_->waitAll(recv_reqs);

    int nrecv = 0;
    for (int n=0; n<n_remote_procs_; ++n)
      nrecv += recv_count[n];

    std::vector<int> recv_start(n_remote_procs_, 0);
    for (int n=1; n<n_remote_procs_; ++n)
      recv_start[n] = recv_start[n-1] + recv_count[n-1];


    // Actually send the particles
    recv_reqs.clear(); send_reqs.clear();
    Kokkos::View<Particle*> recv_buffer("MigrateParticles::recv_buffer", nrecv);
    for (int n=0; n<n_remote_procs_; ++n) {
      if ( recv_count[n] ) {
        Teuchos::ArrayView<char> pr((char*)(recv_buffer.ptr_on_device()+recv_start[n]), recv_count[n]*sizeof(Particle));
        recv_reqs.push_back(particles_.comm_->ireceive(pr, particles_.proc_reference_list_(n)));
      }
      if ( count[n] ) {
        Teuchos::ArrayView<char> ps((char*)(particles_.migrate_particles_.ptr_on_device()+start[n]), count[n]*sizeof(Particle));
        send_reqs.push_back(particles_.comm_->isend(ps, particles_.proc_reference_list_(n)));
      }
    }

    particles_.comm_->waitAll(send_reqs);
    particles_.comm_->waitAll(recv_reqs);

    particles_.cleanup_list();
    particles_.migrate_count_() = 0;

    if (nrecv > 0) {
      for (int i=0; i<nrecv; ++i) {
        recv_buffer(i).ielement = particles_.element_map_->getLocalElement(recv_buffer(i).ielement);
        mesh_.physToReference(recv_buffer(i).x_ref, recv_buffer(i).x, recv_buffer(i).ielement);
        mesh_.physToReferenceVector(recv_buffer(i).v_ref, recv_buffer(i).v, recv_buffer(i).x_ref, recv_buffer(i).ielement);
      }
      UnpackParticlesFromMigration unpack_functor(particles_, recv_buffer, particles_.element_map_);
      int nteams = TeamPolicy::team_size_recommended(unpack_functor);
      TeamPolicy policy((nrecv-1)/nteams+1, nteams);
      Kokkos::parallel_for(policy,  unpack_functor);
      Kokkos::fence();
    }

    particles_.num_received_() = nrecv;
  }
protected:
  ParticleList particles_;
  Mesh mesh_;
  LO n_remote_procs_;
  Kokkos::View<LO*> index_;

};



#endif /* PARTICLE_LIST_FUNCTORS_HPP_ */
