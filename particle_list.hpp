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
 * particle_list.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: mbetten
 */

#ifndef PARTICLE_LIST_HPP_
#define PARTICLE_LIST_HPP_

#include "types.hpp"
#include "Kokkos_View.hpp"
#include "Kokkos_View.hpp"
#include "base_functor.hpp"
#include "particle.hpp"

class DeleteParticles;
class MigrateParticles;
/// @brief a growable container for particles
/// The class holds a series of segmented vectors which hold all the information relating to a particle
/// You have access functions which take a single iterator then you have functions that
/// actually change the list, these must take a member_type because a team must grow/shrink the list
class ParticleList {

public:

  enum KILL_MIGRATE_FLAG {KILL_ONLY = -2, DO_NOTHING };

  /// Constructor
  /// @param max_capacity this is the maximum number of particles which can be stored in this list
  /// @param segment_size this is the size of each segment in this list.
  ParticleList(  std::size_t max_capacity = 32768*1024 );

  void set_comm(Teuchos::RCP< const Teuchos::Comm< int > > comm) {comm_ = comm;}

  /// Access component
  KOKKOS_INLINE_FUNCTION
  GO& ielement  (std::size_t i) const { return ielement_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & x    (std::size_t i) const { return x_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & x_ref(std::size_t i) const { return x_ref_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & v    (std::size_t i) const { return v_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & v_ref(std::size_t i) const { return v_ref_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & E    (std::size_t i) const { return E_(i);}
  KOKKOS_INLINE_FUNCTION
  VECTOR & B    (std::size_t i) const { return B_(i);}
  KOKKOS_INLINE_FUNCTION
  FLOAT & time_remaining (std::size_t i) const { return time_remaining_(i);}
  KOKKOS_INLINE_FUNCTION
  FLOAT & weight(std::size_t i) const { return weight_(i);}
  KOKKOS_INLINE_FUNCTION
  int   & type  (std::size_t i) const { return type_(i);}
  KOKKOS_INLINE_FUNCTION
  dRandom & rand(std::size_t i) const { return rand_(i);}
  KOKKOS_INLINE_FUNCTION
  bool & is_dead(std::size_t i) const { return is_dead_(i);}

  void set_time_remaining(FLOAT tr) {
    Kokkos::deep_copy(time_remaining_,tr);
    Kokkos::fence();
  }

  /// Particle pushback function
  /// @param member_type this is team specs, the full team needs to call this function
  /// @param p this is an array of particles we are adding, one per team member
  /// @param add this is a bool array if we are going to add the particle in p
  /// @param list_size size of the p/add array
  KOKKOS_INLINE_FUNCTION
  void push_back(const member_type  &dev, const Particle *p, long *add, int list_size) const  {
    const int team_rank = dev.team_rank();

    dev.team_barrier();
    std::size_t grow_size=0;
    for (int i=0; i<list_size; ++i)
      if ( add[i] )
        ++ grow_size;
    
    if ( team_rank == 0 ) {
      std::size_t final_size = Kokkos::atomic_fetch_add(&count_(),grow_size);
      for (int i=0; i<list_size; ++i)
        if ( add[i] )
          add[i] = final_size++;
        else
          add[i] = -1;
      assert ( final_size < is_dead_.dimension_0());
    }
    dev.team_barrier();

    if ( add[team_rank] >= 0 ) {
      ielement_(add[team_rank]) = p[team_rank].ielement;
      x_(add[team_rank]) = p[team_rank].x;
      x_ref_(add[team_rank]) = p[team_rank].x_ref;
      v_(add[team_rank]) = p[team_rank].v;
      v_ref_(add[team_rank]) = p[team_rank].v_ref;
      E_(add[team_rank]) = p[team_rank].E;
      B_(add[team_rank]) = p[team_rank].B;
      time_remaining_(add[team_rank]) = p[team_rank].time_remaining;
      weight_(add[team_rank]) = p[team_rank].weight;
      type_(add[team_rank]) = p[team_rank].type;
      rand_(add[team_rank]) = p[team_rank].rand;
      is_dead_(add[team_rank]) = p[team_rank].is_dead;
    }
    dev.team_barrier();
  }

  /// Mark a particle for deletion, this does not actually delete a particle, @sa cleanup_list removes the memory
  /// @param dev team member for deleting a particle
  /// @param i index of particle to add to delete list
  KOKKOS_INLINE_FUNCTION
  void delete_particle(const member_type &dev, std::size_t *i, std::size_t num_deleted) const  {
    const int team_rank = dev.team_rank();

    if ( team_rank == 0 ) {
      std::size_t deleted_index = Kokkos::atomic_fetch_add(&deleted_count_(),num_deleted);
      assert( deleted_index + num_deleted < deleted_particle_list_.dimension_0());

      for ( std::size_t indx=0; indx< num_deleted; ++ indx)
        deleted_particle_list_(deleted_index+indx) = i[indx];
    }
    dev.team_barrier();
  }

  /// This cleans up the deleted particles and moves the last set of particles to the new empty slots.
  /// THis needs to be called from the host
  void cleanup_list() const;

  /// Returns the number of particles int he list
  /// Needs to be called from the host
  KOKKOS_INLINE_FUNCTION
  std::size_t num_particles()  const{
    return count_();
  }

  std::size_t num_migrate()  const{
    return migrate_count_();
  }


  /// Add particles to the list for migration
  /// @param dev team specification
  /// @param particle_id_list list of particles to add to the migrate list
  /// @param proc_list list of processors to migrate to, -1 for not migrating that particle
  KOKKOS_INLINE_FUNCTION
  void add_particles_to_migrate_list(const member_type &dev, GO *particle_id_list, int *proc_list, int cnt) const {

    const LO team_rank = dev.team_rank();

    dev.team_barrier();
    int n_migrate = 0, n_kill=0;;
    for (int iteam=0; iteam< cnt; ++iteam) {
      if (proc_list[iteam] > ParticleList::DO_NOTHING)
        ++n_migrate;
      if (proc_list[iteam] == ParticleList::KILL_ONLY)
        ++n_kill;
    }
    if ( n_migrate+n_kill == 0 )
      return;

    n_kill += n_migrate;
    if (team_rank == 0) {
      std::size_t current_size = Kokkos::atomic_fetch_add(&migrate_count_(),n_migrate);
      assert( current_size+n_migrate < migrate_list_.dimension_0());

      std::size_t current_delete_size = Kokkos::atomic_fetch_add(&deleted_count_(),n_kill);
      assert(current_delete_size +n_kill < deleted_particle_list_.dimension_0());

      for (int iteam=0; iteam< cnt; ++iteam)
        if (proc_list[iteam] > -1) {
          migrate_list_(current_size++) = Kokkos::pair<GO, int>( particle_id_list[iteam], proc_list[iteam]);
          deleted_particle_list_(current_delete_size++) = particle_id_list[iteam];
          is_dead_(particle_id_list[iteam]) = true;
        } else if (proc_list[iteam] == ParticleList::KILL_ONLY) {
          deleted_particle_list_(current_delete_size++) = particle_id_list[iteam];
          is_dead_(particle_id_list[iteam]) = true;
        }
    }
    dev.team_barrier();
  }

  /// This function migrates all the particles on the particle list
  /// and returns true if particles were on the list, false if not
  bool perform_migrate(MigrateParticles &migrator) const;

  /// write all the particles to out
  /// called from the host
  void dump_particles(std::ostream &out) const {
    int i, n = num_particles();
    for (i=0; i<n; ++i)
      out << ielement_(i) << "("<<x_(i)[0]<<", "<<x_(i)[1]<<", "<<x_(i)[2] <<") "
      << "("<<v_(i)[0]<<", "<<v_(i)[1]<<", "<<v_(i)[2] <<") "<<type_(i)<<"\n";
  }

  /// Returns the max capacity of the list
  GO max_capacity() {return max_capacity_;}

  /// Returns the current capacity
  KOKKOS_INLINE_FUNCTION
  GO current_capacity() { return ielement_.size();}

  /// Set up the processor communication pattern
  void set_neighboring_procs(std::set<int> neighbors);

  /// Sets the map for elements on the particle mesh for migration of particles
  void set_element_map(Teuchos::RCP<Map> map) { element_map_ = map;}

  /// Sort the particle list
  void sort();

  /// Create a start index for the different elements, calls sort
  void compute_start_particles_by_element();

  struct ComputeStartParticlesByElementFunctor{};
  KOKKOS_INLINE_FUNCTION
  void operator() (ComputeStartParticlesByElementFunctor, LO elem) const;

  /// Returns the start position for a particle in an element
  KOKKOS_INLINE_FUNCTION
  LO start(LO elem) const { return start_by_element_(elem);}

  /// returns the count by element
  KOKKOS_INLINE_FUNCTION
  LO count_for_element(LO elem) const { return start_by_element_(elem+1)-start_by_element_(elem);}

  /// Return a particle based on an index
  KOKKOS_INLINE_FUNCTION
  Particle particle(std::size_t i) const {
    Particle p;
    p.x = x_(i);
    p.x_ref = x_ref_(i);
    p.v = v_(i);
    p.v_ref = v_ref_(i);
    p.E = E_(i);
    p.B = B_(i);
    p.ielement = ielement_(i);
    p.time_remaining = time_remaining_(i);
    p.weight = weight_(i);
    p.type = type_(i);
    p.rand = rand_(i);
    p.is_dead = is_dead_(i);
    return p;
  }

protected:

  Teuchos::RCP< const Teuchos::Comm< int > > comm_;

  Kokkos::View<GO*, DeviceSpace> ielement_;
  Kokkos::View<VECTOR*, DeviceSpace> x_;
  Kokkos::View<VECTOR*, DeviceSpace> x_ref_;
  Kokkos::View<VECTOR*, DeviceSpace> v_;
  Kokkos::View<VECTOR*, DeviceSpace> v_ref_;
  Kokkos::View<VECTOR*, DeviceSpace> E_;
  Kokkos::View<VECTOR*, DeviceSpace> B_;
  Kokkos::View<FLOAT*, DeviceSpace> time_remaining_;
  Kokkos::View<FLOAT*, DeviceSpace> weight_;
  Kokkos::View<int*, DeviceSpace> type_;
  Kokkos::View<dRandom*, DeviceSpace> rand_;
  Kokkos::View<bool*, DeviceSpace> is_dead_;

  Kokkos::View<LO*, DeviceSpace> start_by_element_;

  Kokkos::View<std::size_t*, DeviceSpace> deleted_particle_list_;

  Kokkos::View<std::size_t> count_;
  Kokkos::View<std::size_t> deleted_count_;

  Kokkos::View<Kokkos::pair<GO, int>*, DeviceSpace> migrate_list_;
  Kokkos::View<std::size_t, DeviceSpace> migrate_count_;

  Kokkos::View<Particle*, DeviceSpace> migrate_particles_;
  Kokkos::View<LO *, DeviceSpace> proc_reference_list_;

  Kokkos::View<GO> num_received_;

  GO max_capacity_;

  Teuchos::RCP<Map> element_map_;

  friend class DeleteParticles;
  friend class MigrateParticles;
  /// THis is a unit test
  friend struct TestMigrationSetup;

};

#include "particle_list_functors.hpp"


#endif /* PARTICLE_LIST_HPP_ */
