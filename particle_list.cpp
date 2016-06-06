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
 * particle_list.cpp
 *
 *  Created on: Oct 9, 2014
 *      Author: mbetten
 */



#include "particle_list.hpp"
#include<Kokkos_Sort.hpp>

ParticleList::ParticleList(  std::size_t max_capacity):
  ielement_("ParticleList::ielement", max_capacity),
  x_("ParticleList::x", max_capacity),
  x_ref_("ParticleList::x_ref", max_capacity),
  v_("ParticleList::v", max_capacity),
  v_ref_("ParticleList::v_ref", max_capacity),
  E_("ParticleList::E", max_capacity),
  B_("ParticleList::B", max_capacity),
  time_remaining_("ParticleList::time_remaining",max_capacity),
  weight_("ParticleList::weight", max_capacity),
  type_("ParticleList::type", max_capacity),
  rand_("ParticleList::rand", max_capacity),
  is_dead_("ParticleList::is_dead", max_capacity),
  start_by_element_("ParticleList::start_by_element_", 0),
  deleted_particle_list_("ParticleList::deleted_particle_list", max_capacity),
  count_("CountOfParticles"),deleted_count_("DeletedCountOfParticles"),
  migrate_list_("ParticleList::migrate_list", max_capacity),
  migrate_count_("CountOfMigrateList"),
  migrate_particles_("ParticleList::migrate_particles"),
  proc_reference_list_("ParticleList::proc_reference_list"),
  num_received_("ParticleList::num_received"),
  max_capacity_(max_capacity)
{

}

void ParticleList::cleanup_list() const {
  DeleteParticles deleter(*this);
  deleter.execute();
}

void ParticleList::set_neighboring_procs(std::set<int> neighbors){
   int num_neighbors = neighbors.size();
   proc_reference_list_ = Kokkos::View<LO*>("ParticleList::proc_reference_list", num_neighbors);
   std::set<int>::iterator iter = neighbors.begin(), end = neighbors.end();
   for (int cnt=0; iter != end; ++iter)
     proc_reference_list_(cnt++) = *iter;
 }

bool ParticleList::perform_migrate(MigrateParticles &migrator) const {
  migrator.execute();
  GO total_parts = 0;
  Teuchos::reduceAll(*comm_, Teuchos::REDUCE_SUM, 1, &num_received_(), &total_parts);
  return total_parts > 0;
}

void ParticleList::sort() {
  typedef Kokkos::View<GO*, DeviceSpace> KeyViewType;
  typedef Kokkos::SortImpl::DefaultBinOp1D< KeyViewType > BinOp;

  int max_elems = element_map_->getNodeNumElements(), min_elems = 0, num_elems = max_elems-min_elems+1;
  BinOp binner(num_elems, min_elems, max_elems);

  Kokkos::pair<size_t, size_t> reduced_size(0,num_particles());
  KeyViewType local_ielement = Kokkos::subview(ielement_, reduced_size);

  Kokkos::BinSort< KeyViewType , BinOp > Sorter(local_ielement,binner,false);
  Sorter.create_permute_vector();
  Sorter.sort< KeyViewType >(local_ielement);

  Sorter.sort(Kokkos::subview(x_, reduced_size));
  Sorter.sort(Kokkos::subview(x_ref_, reduced_size));
  Sorter.sort(Kokkos::subview(v_, reduced_size));
  Sorter.sort(Kokkos::subview(v_ref_, reduced_size));
  Sorter.sort(Kokkos::subview(E_, reduced_size));
  Sorter.sort(Kokkos::subview(B_, reduced_size));
  Sorter.sort(Kokkos::subview(time_remaining_, reduced_size));
  Sorter.sort(Kokkos::subview(weight_, reduced_size));
  Sorter.sort(Kokkos::subview(type_, reduced_size));
  Sorter.sort(Kokkos::subview(rand_, reduced_size));
  Sorter.sort(Kokkos::subview(is_dead_, reduced_size));

}

void ParticleList::operator() (ComputeStartParticlesByElementFunctor, LO elem) const {
  LO n_part = count_();
  LO low = 0, hi = n_part, mid = (low+hi)/2;
  while (hi-low > 1) {
    if ( ielement_(mid) < elem)
      low = mid;
    else
      hi = mid;
    mid = (low+hi)/2;
  }
  if (ielement(mid) >= elem)
    start_by_element_(elem) = mid;
  else
    start_by_element_(elem) = hi;

}

void ParticleList::compute_start_particles_by_element() {

  LO count = element_map_->getNodeNumElements()+1; // goes from 0->n-1, but we need counts to n so we can go count[n]-count[n-1]
  if ( count != static_cast<LO>(start_by_element_.dimension_0()))
    start_by_element_ = Kokkos::View<LO*, DeviceSpace>("ParticleList::start_by_element_", count);
  Kokkos::fence();
  Kokkos::deep_copy(start_by_element_, num_particles());
  Kokkos::fence();
  sort();
  Kokkos::fence();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceSpace,ComputeStartParticlesByElementFunctor>(0,count-1), *this);
}

void DeleteParticles::execute() {
  num_particles_ = particles_.num_particles();
  num_deleted_ = particles_.deleted_count_();
  Kokkos::deep_copy(fill_from_, num_particles_ - 1);

  // Reset the size of the list
  Kokkos::fence();
  num_particles_ -= num_deleted_;
  particles_.count_() -= num_deleted_;
  Kokkos::fence();
  if (num_particles_ >  0) {  // not deleting the whole list
    Kokkos::parallel_for(num_deleted_, *this);
  }

  Kokkos::fence();
  particles_.deleted_count_() = 0;
}

/// Kokkos functor to delete particle
KOKKOS_INLINE_FUNCTION
void DeleteParticles::operator() (std::size_t i) const{
  std::size_t index = particles_.deleted_particle_list_(i);
  if ( index >= num_particles_)
    return;

  LO get_from = Kokkos::atomic_fetch_add(&fill_from_(), -1);
  while (particles_.is_dead(get_from ) ) {
    // We need to grab one from the end of the list instead
    get_from = Kokkos::atomic_fetch_add(&fill_from_(), -1);
  }
  particles_.x(index) = particles_.x(get_from );
  particles_.x_ref(index) = particles_.x_ref(get_from );
  particles_.v(index) = particles_.v(get_from );
  particles_.v_ref(index) = particles_.v_ref(get_from );
  particles_.E(index) = particles_.E(get_from );
  particles_.B(index) = particles_.B(get_from );
  particles_.time_remaining(index) = particles_.time_remaining(get_from );
  particles_.weight(index) = particles_.weight(get_from );
  particles_.type(index) = particles_.type(get_from );
  particles_.rand(index) = particles_.rand(get_from );
  particles_.ielement(index) = particles_.ielement(get_from );
  particles_.is_dead(index) = particles_.is_dead(get_from );

}

