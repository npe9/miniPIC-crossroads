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
 * particle_fill.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: mbetten
 */

#ifndef PARTICLE_FILL_HPP_
#define PARTICLE_FILL_HPP_

#include "base_functor.hpp"
#include "mesh.hpp"
#include "particle_list.hpp"
#include "data_warehouse.hpp"
#include <cstdlib>



/// Class to fill a mesh with a constant number of particles in each element.
/// this will lead to an uneven density since it doens't take the element volume
/// into account
class ParticleFill {
public:
  /// constructor
  /// @param mesh input mesh to fill
  /// @param parts input list of particles to fill
  /// @param num_parts_per_element number of particles to insert into every element
  ParticleFill ( DataWarehouse &data, int num_parts_per_element, int type, FLOAT density, VECTOR vel = VECTOR(3,0), FLOAT temp=1.0)
  : data_(data), particles_(data.particles()), mesh_(data.mesh()) {
    if ( particles_.max_capacity() < static_cast<GO>(mesh_.num_elems*num_parts_per_element) ) {
      std::cout << "Cannot create that number of particles, no fill performed\n";
      return;
    }

    FillFunctor func(mesh_, particles_, data_.particle_type_list(), num_parts_per_element, type, density, vel, temp);
    func.execute();
  }


public:
  /// Kokkos functor to fill the mesh with the particles
  class FillFunctor: public BaseKokkosFunctor {
  public:
    typedef DeviceSpace::scratch_memory_space shmem_space ;

    FillFunctor(const Mesh &mesh, ParticleList &particles, ParticleTypeList particle_types, size_t num_parts_per_element, int type, FLOAT density, VECTOR vel, FLOAT temp) :
      BaseKokkosFunctor("FillFunctor"),
      mesh_(mesh), particles_(particles), particle_types_(particle_types), num_parts_per_element_(num_parts_per_element), type_(type), density_(density), vel_(vel), temp_(temp)
       { }

    /// Function to execute functor from the host
    void execute() {
      std::size_t num = mesh_.num_elems;
      unsigned int team_recommended = std::min<size_t>(TeamPolicy::team_size_recommended(*this),num_parts_per_element_) ;
      TeamPolicy policy(num, team_recommended);
      Kokkos::parallel_for(policy, *this);
      Kokkos::fence();
    }

    /// This function says how much shared memory to allocate
    /// needed for the functor
    unsigned team_shmem_size(int team_size) const {
      return Kokkos::View<Particle* , shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size) +
        Kokkos::View<long*, shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size) ;
    }

  protected:
    /// Compute vel from average and temp
    KOKKOS_INLINE_FUNCTION
    void set_vel(Particle &p) const {
      if ( temp_ == 0) {
        p.v = vel_;
        return;
      }
      if (temp_ < 0 ) {
        // This is 2stream
        p.v[0] = vel_[0]*(1.+temp_*sin(1.*M_PI*p.x[0]));
        p.v[1] = p.v[2] = 0.0;
        return;
      }
      do {
        FLOAT r = p.rand.drand();
        FLOAT vmax = sqrt(6*temp_/particle_types_.particle_info(p.type).m);
        VECTOR v;
        v[0] = vmax*2*(-.5+p.rand.drand());v[1] = vmax*2*(-.5+p.rand.drand());v[2] = vmax*2*(-.5+p.rand.drand());
        FLOAT prob = exp(-(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])/temp_);
        if ( r < prob ) {
          p.v[0] = vel_[0]+v[0];p.v[1] = vel_[1]+v[1];p.v[2] = vel_[2]+v[2];
          break;
        }
      } while (true);

    }


  public:
    /// Fill functor
    KOKKOS_INLINE_FUNCTION
    void operator () (const member_type & dev) const {
      const GO n_global_elem = mesh_.num_global_elems;
      const LO elem = dev.league_rank();
      const LO team_size = dev.team_size();
      const LO team_rank = dev.team_rank();
      ConstLenVector<FLOAT, 3> ref, phys;
      Kokkos::View<Particle*, shmem_space, Kokkos::MemoryUnmanaged> p(dev.team_shmem(), team_size);
      Kokkos::View<long*, shmem_space, Kokkos::MemoryUnmanaged> is_filled(dev.team_shmem(), team_size);

      FLOAT vol = mesh_.determinate_jacobian(elem)/6.; // tet vol is 1/6 so det needs to be normalized
      p(team_rank).ielement = elem;
      p(team_rank).type = type_;
      p(team_rank).weight = density_*vol/num_parts_per_element_;
      p(team_rank).is_dead = false;

      const size_t end = num_parts_per_element_+team_rank;
      for (size_t ipart= team_rank; ipart < end; ipart += team_size){
        p(team_rank).rand.srand(elem + (34030+ipart)*n_global_elem + n_global_elem*12345*type_);
        for (int i=0; i<10; ++i) // Burn the first 10 rands
          p(team_rank).rand.rand();
        is_filled(team_rank) = false;
        if ( ipart < num_parts_per_element_) {
          do {
            p(team_rank).x_ref[0] = p(team_rank).rand.drand();
            p(team_rank).x_ref[1] = p(team_rank).rand.drand();
            p(team_rank).x_ref[2] = p(team_rank).rand.drand();
          } while (p(team_rank).x_ref[0]+p(team_rank).x_ref[1]+p(team_rank).x_ref[2] > 1.0);
          mesh_.refToPhysical(p(team_rank).x, p(team_rank).x_ref, elem);
          set_vel(p(team_rank));
          mesh_.physToReferenceVector(p(team_rank).v_ref, p(team_rank).v, p(team_rank).x_ref, elem);
          is_filled(team_rank) = true;
        }
        particles_.push_back(dev, &p(0), &is_filled(0), team_size);
      }
    }

    friend class ParticleFill;

  protected:
    const Mesh mesh_;
    ParticleList particles_;
    size_t num_parts_per_element_;
    int type_;
    FLOAT density_;
    VECTOR vel_;
    FLOAT temp_;
    ParticleTypeList particle_types_;
  };
  protected:
    DataWarehouse &data_;
    ParticleList particles_;
    const Mesh mesh_;

};

#endif /* PARTICLE_FILL_HPP_ */
