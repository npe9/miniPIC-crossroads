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
 * particle_move.hpp
 *
 *  Created on: Sep 15, 2014
 *      Author: mbetten
 */

#ifndef PARTICLE_MOVE_HPP_
#define PARTICLE_MOVE_HPP_


#include "particle_list.hpp"
#include "particle_type.hpp"
#include "mesh.hpp"
#include "base_functor.hpp"
#include "utils.hpp"
#include "data_warehouse.hpp"

/// This class moves the particles in a list
/// It is a compound class which has several functor classes in it
/// which actually move the particle.
class ParticleMove {
public:


  /// Class to move the particle in the mesh, it computes element crossing and migrates particles
  /// from element to element, it has hard coded reflection off mesh boundaries and does not work
  /// yet in parallel
  class MoveParticle: public BaseKokkosFunctor {
    public:
        typedef DeviceSpace::scratch_memory_space shmem_space ;

        MoveParticle(const Mesh &mesh, const ParticleList &particles, bool kill_particles = false) :
        BaseKokkosFunctor("ParticleListFillFunctor"),
        mesh_(mesh),  elem_map_(mesh_.element_map->getLocalMap()),particles_(particles), dt_(0), num_particles_(-1),
        global_working_particle_("working_particle"),kill_particle_on_wall_(kill_particles)
        {
        }

      /// Function to execute functor from the host
      void execute(double dt)  {
        MigrateParticles migrator(particles_, mesh_);
        dt_ = dt;
        num_particles_ = particles_.num_particles();
        unsigned int team_size = TeamPolicy::team_size_recommended(*this) ;
        particles_.set_time_remaining(1.0);
        bool cont = true;
        Kokkos::deep_copy(global_working_particle_, 0);
        Kokkos::fence();
        LO num_teams = ceil(num_particles_*1./team_size);
        do {
          Kokkos::parallel_for(TeamPolicy(num_teams, team_size), *this);
          Kokkos::fence();
          LO last_completed_particle = particles_.num_particles() - particles_.num_migrate();
          Kokkos::deep_copy(global_working_particle_, last_completed_particle);
          Kokkos::fence();
          cont = particles_.perform_migrate(migrator);
          num_particles_ = particles_.num_particles();
          num_teams = ceil((num_particles_-last_completed_particle)*1./team_size);
        } while (cont);
      }
      /// This function says how much shared memory to allocate
      /// needed for the functor
      unsigned team_shmem_size(int team_size) const {
        return
             Kokkos::View<GO* , shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size) +
             Kokkos::View<int*, shmem_space, Kokkos::MemoryUnmanaged>::shmem_size(team_size) +
            3*Kokkos::View<int, shmem_space, Kokkos::MemoryUnmanaged>::shmem_size() ;
      }

      KOKKOS_INLINE_FUNCTION
      void operator() (const member_type & dev) const {
        const LO league_rank = dev.league_rank();
        const LO team_size = dev.team_size();
        const LO team_rank = dev.team_rank();
        Kokkos::View<GO*, shmem_space, Kokkos::MemoryUnmanaged> migrate_list(dev.team_shmem(), team_size);
        Kokkos::View<int*, shmem_space, Kokkos::MemoryUnmanaged> proc_list(dev.team_shmem(), team_size);

        GO i = league_rank*team_size+team_rank + global_working_particle_();
        proc_list(team_rank) = -1;
        if ( i < num_particles_ && !particles_.is_dead(i)) {
          while (particles_.time_remaining(i) > 0 ){
            particles_.time_remaining(i) -= find_intersection(i, proc_list(team_rank), kill_particle_on_wall_) *
                particles_.time_remaining(i);
            if ( proc_list(team_rank) > ParticleList::DO_NOTHING || proc_list(team_rank) == ParticleList::KILL_ONLY) {
              migrate_list(team_rank) = i;
              break;
            }
          }
        }
        particles_.add_particles_to_migrate_list(dev, &migrate_list(0), &proc_list(0), team_size);

      }

      void reset_particle_list(ParticleList &parts) {
         particles_ = parts;
       }

    protected:
      /// This routine computes what fraction of the timestep until a particle hits an element boundary
      /// The algorithm computes an average reference velocity which places the particle at the same point
      /// at the end of fractional timestep in both reference and physical space.
      /// This only works for TET meshes
      KOKKOS_INLINE_FUNCTION
      FLOAT find_intersection(std::size_t i, int &neighbor_proc, bool kill_particle_on_wall=false) const {
        VECTOR dx, x_ref_final;
        VECTOR &x = particles_.x(i), &x_ref = particles_.x_ref(i);
        VECTOR &v = particles_.v(i), &v_ref = particles_.v_ref(i);
        GO &ielement = particles_.ielement(i);
        const FLOAT local_dt = dt_*particles_.time_remaining(i);
        FLOAT scale_fact = 1.0;
        int face = -1;
        neighbor_proc = -1;
        face = -1;
        scale_fact = 1.0;
        VECTOR dx_ref;
        for (int idim=0; idim<3; ++idim) {
          dx_ref[idim] = local_dt*v_ref[idim];
          x_ref_final[idim] = x_ref[idim] + dx_ref[idim];
        }
        if ( v_ref[2] < 0 && x_ref_final[2] < 0 ) {
          face = 3;
          scale_fact = -x_ref[2]/(dx_ref[2]);
        }
        if( v_ref[1] < 0 && x_ref_final[1] < 0 ) {
          FLOAT sfy;
          sfy = -x_ref[1]/(dx_ref[1]);
          if ( sfy < scale_fact) {
            face = 0;
            scale_fact = sfy;
          }
        }

        if( v_ref[0] < 0 && x_ref_final[0] < 0 ) {
          FLOAT sfx;
          sfx = -x_ref[0]/(dx_ref[0]);
          if ( sfx < scale_fact) {
            face = 2;
            scale_fact = sfx;
          }
        }
        FLOAT x_ref_final_sum = x_ref_final[0] + x_ref_final[1] +  x_ref_final[2];
        FLOAT v_ref_sum = v_ref[0] + v_ref[1] +  v_ref[2];
        if( v_ref_sum > 0 && x_ref_final_sum > 1 ) {
          FLOAT sfs;
          sfs = (1-x_ref[0]-x_ref[1]-x_ref[2])/(dx_ref[0]+dx_ref[1]+dx_ref[2]);
            if ( sfs < scale_fact) {
              face = 1;
              scale_fact = sfs;
            }
        }
        for (int idim=0; idim<3; ++idim) {
          x[idim] = x[idim] +scale_fact*local_dt*v[idim] ;
          x_ref[idim] = x_ref[idim] +scale_fact*local_dt*v_ref[idim] ;
        }
        if ( scale_fact < 1 ) {  // Hit something
          GO mapped_element = mesh_.elem_face_to_elem_gids(ielement, face);
          if (mapped_element >= 0 ) { // didn't hit a wall, -1 mean a wall
            if ( mesh_.is_periodic && mesh_.elem_face_periodic(ielement, face) != 0 )
              x[0] = -x[0];
            int remote_rank = mesh_.elem_face_to_proc(ielement, face);
	    //#ifdef KOKKOS_HAVE_CUDA
	    //            LO new_element = mapped_element;
	    //#else

	    LO new_element = elem_map_.getLocalElement(mapped_element);
	    //#endif
            if ( new_element < 0 ) {
              neighbor_proc = mesh_.elem_face_to_proc(ielement, face);
              ielement = mapped_element;
	      assert(mapped_element >= 0);
            }else {
              ielement=new_element;
              mesh_.physToReference(x_ref, x, ielement);
              mesh_.physToReferenceVector(v_ref, v, x_ref, ielement);
            }
          } else { // reflection or Kill
            if (kill_particle_on_wall) {
              neighbor_proc = ParticleList::KILL_ONLY;
              return 1;
            }
            VECTOR &normal = mesh_.normals(ielement, face);
            FLOAT v_dot_n=0;
            for (int idim=0; idim<3; ++idim)
              v_dot_n += v[idim]*normal[idim];
            for (int idim=0; idim<3; ++idim)
              v[idim] -= 2*v_dot_n*normal[idim];

            mesh_.physToReferenceVector(v_ref, v, x_ref, ielement);
          }
        }
        assert(scale_fact > 0);
        return scale_fact;
      }

      /// This routine computes what fraction of the timestep until a particle hits an element boundary
      /// The algorithm computes an average reference velocity which places the particle at the same point
      /// at the end of fractional timestep in both reference and physical space.
      /// This only works for HEX meshes
      KOKKOS_INLINE_FUNCTION
      FLOAT find_intersection_hex(std::size_t i, int &neighbor_proc, bool kill_particle_on_wall=false) const {
        VECTOR dx, x_ref_final, x_final, old_x_final, x_from_ref_final;
        VECTOR &x = particles_.x(i), &x_ref = particles_.x_ref(i);
        VECTOR &v = particles_.v(i), &v_ref = particles_.v_ref(i);
        GO &ielement = particles_.ielement(i);
        FLOAT local_dt = dt_;
        FLOAT scale_fact = 1.0;
        int face = -1;
        const int front_face[3] = {1, 2, 5}, back_face[3]={3, 0, 4};
        const int niter_max=10;
        neighbor_proc = -1;
        for (int niter = 0 ; niter <= niter_max; ++niter) {
          face = -1;
          scale_fact = 1.0;
          for (int idim=0; idim<3; ++idim) {
            x_final[idim] =  x[idim] + local_dt*particles_.time_remaining(i)*v[idim];
            FLOAT dx_ref = local_dt*particles_.time_remaining(i)*v_ref[idim];
            x_ref_final[idim] = x_ref[idim] + dx_ref;
            if ( x_ref_final[idim] > 1.0 && dx_ref > 0) {
              FLOAT s =  (+1.- x_ref[idim]) / dx_ref;
              if ( s < scale_fact) {
                scale_fact = s;
                face = front_face[idim];
              }
            }else if ( x_ref_final[idim] < -1.0 && dx_ref < 0) {
              FLOAT s =(-1.- x_ref[idim]) / dx_ref;
              if ( s < scale_fact) {
                scale_fact = s;
                face = back_face[idim];
              }
            }
          }

          for (int idim=0; idim<3; ++idim) {
            x_final[idim] =  x[idim] + scale_fact*local_dt*particles_.time_remaining(i)*v[idim];
            x_ref_final[idim] = x_ref[idim] + scale_fact*local_dt*particles_.time_remaining(i)*v_ref[idim];
          }
          mesh_.refToPhysical(x_from_ref_final, x_ref_final, ielement);
          for (int idim=0; idim<3; ++idim)
            dx[idim] = x_final[idim] - x_from_ref_final[idim];
          if ( utils::dot(dx, dx) < 1e-12 || niter == niter_max ) {
            for (int idim=0; idim<3; ++idim) {
              x[idim] = x_final[idim];
              x_ref[idim] = x_ref_final[idim];
            }
            VECTOR x_check;
            if ( scale_fact < 1 ) {
              GO mapped_element = mesh_.elem_face_to_elem_gids(ielement, face);
              if (mapped_element >= 0 ) {
                if (face == 0)
                  x_ref[1] = 1;
                else if (face == 1)
                  x_ref[0] = -1;
                else if (face == 2)
                  x_ref[1] = -1;
                else if (face == 3)
                  x_ref[0] = 1;
                else if (face == 4)
                  x_ref[2] = 1;
                else if (face == 5)
                  x_ref[2] = -1;
                int remote_rank = mesh_.elem_face_to_proc(ielement, face);
#ifdef KOKKOS_HAVE_CUDA
                ielement = mapped_element;
#else
                ielement = mesh_.element_map->getLocalElement(mapped_element);
#endif
                if ( ielement < 0 ) {
                  neighbor_proc = remote_rank;
                  ielement = mapped_element;
                }else {
                  mesh_.physToReferenceVector(v_ref, v, x_ref, ielement);
                }
              } else { // reflection or Kill
                if (kill_particle_on_wall) {
                  neighbor_proc = ParticleList::KILL_ONLY;
                  return 1;
                }
                if (face == 0 ||face == 2)
                  v[1] *= -1;
                else if (face == 1 || face == 3)
                  v[0] *= -1;
                else if (face == 4 ||face == 5)
                  v[2] *= -1;
                mesh_.physToReferenceVector(v_ref, v, x_ref, ielement);
                for (int idim=0; idim<3; ++idim)
                  if ( x_ref[idim] == -1 )
                    v_ref[idim] = utils::abs(v_ref[idim]);
                  else if ( x_ref[idim] == 1 )
                    v_ref[idim] = -utils::abs(v_ref[idim]);
              }
            }
            return scale_fact;
          }
          // We didn't converge so reset v_ref
          VECTOR x_ref_midpoint, v_correction, v_ref_correction;
          for (int idim=0; idim<3; ++idim) {
            v_correction[idim] = dx[idim] / (scale_fact*local_dt);
            x_ref_midpoint[idim] = 0.5*(x_ref[idim]+x_ref_final[idim]);
          }
          mesh_.physToReferenceVector(v_ref_correction, v_correction, x_ref_midpoint, ielement);
          for (int idim=0; idim<3; ++idim) 
            v_ref[idim] += v_ref_correction[idim];
        }
        return scale_fact;
      }

    protected:
      const Mesh mesh_;
    const Map::local_map_type elem_map_;
      ParticleList particles_;
      double dt_;
      GO num_particles_;
      Kokkos::View<int> global_working_particle_;
      bool kill_particle_on_wall_;
    };



  ParticleMove ( DataWarehouse &data) : data_(data), mesh_(data_.mesh()),particles_(data_.particles()), types_(data.particle_type_list()),
      move_particle_(mesh_, particles_, data_.kill_particles_on_wall_impact()){
    particles_.set_neighboring_procs(mesh_.neighboring_procs);
    particles_.set_element_map(mesh_.element_map);
    data.particles_reference() = particles_;
    move_particle_.reset_particle_list(particles_);
    scale_ = 1.0;
  }

  typedef DeviceSpace::scratch_memory_space shmem_space ;
  struct AccelParticlesTag {};


  KOKKOS_INLINE_FUNCTION
  void operator() (AccelParticlesTag, std::size_t i) const {
    FLOAT scaled_dt_=dt_*scale_;
    VECTOR &v = particles_.v(i);
    VECTOR &E = particles_.E(i);
    VECTOR &B = particles_.B(i);
    VECTOR t;
    FLOAT q_over_m = types_.particle_info(particles_.type(i)).q_over_m;
    t[0] = q_over_m*0.5*B[0];
    t[1] = q_over_m*0.5*B[1];
    t[2] = q_over_m*0.5*B[2];

    for (int idim=0; idim<3; idim++)
      v[idim] += scaled_dt_*E[idim]*q_over_m*0.5;

    VECTOR tmp(v);
    utils::add_cross(tmp, v, t);
    double t_sqr = utils::dot(t,t);
    for (int idim=0; idim<3; idim++)
      t[idim] *= 2./(1+t_sqr);
    utils::add_cross(v, tmp, t);

    for (int idim=0; idim<3; idim++)
      v[idim] += scaled_dt_*E[idim]*q_over_m*0.5;

    mesh_.physToReferenceVector(particles_.v_ref(i) , v, particles_.x_ref(i), particles_.ielement(i));


  }




  /// Move the particles.  currently this doesn't work in parallel
  void move(double dt) {
    dt_ = dt;
    Kokkos::fence();
    parallel_for(Kokkos::RangePolicy<DeviceSpace,AccelParticlesTag >(0,particles_.num_particles()), *this);
    Kokkos::fence();
    move_particle_.execute(dt);
  }

  void set_force_scale(FLOAT scale=1) {
    scale_ = scale;
  }
protected:


  /// Take an index for a particle and push it to the migrate list
  /// @param which_part the index of the particle to migrate
  /// Need to be called on the device
  KOKKOS_INLINE_FUNCTION
  void packForMigrate(std::size_t which_part);

  /// Actually migrate the particles between the different particles.
  /// Needs to be called from the host
  void migrate();

  DataWarehouse &data_;
  const Mesh mesh_;
  ParticleList particles_;
  ParticleTypeList types_;

  // Functor classes
  MoveParticle move_particle_;

  FLOAT dt_;
  FLOAT scale_;
  GO num_particles_;
  Kokkos::View<int> global_working_particle_;
  bool kill_particle_on_wall_;

};



#endif /* PARTICLE_MOVE_HPP_ */
