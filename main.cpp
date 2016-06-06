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
#include <mpi.h>
#include "Kokkos_Core.hpp"
#include <impl/Kokkos_Timer.hpp>
#include "Teuchos_GlobalMPISession.hpp"
#include "types.hpp"
#include "mesh.hpp"
#include "particle_list.hpp"
#include "particle_move.hpp"
#include "particle_fill.hpp"
#include "base_functor.hpp"
#include "data_warehouse.hpp"
#include "ES.hpp"
#include "PIC.hpp"

#include "Teuchos_CommandLineProcessor.hpp"

#include <limits>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>
using namespace std;


int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  Teuchos::RCP<Teuchos::Comm<int> >  comm(new Teuchos::MpiComm<int>(MPI_COMM_WORLD) );

  std::string meshfile("brick.txt");
  std::string problem("none");
  int num_particles_per_element=10000;
  double dt=0.1;
  double t_final = 2.0;
  FLOAT density =  2./3.*M_PI*M_PI;
  int part_kill=0;
  int output=0;
  int dump_mesh= false;
  Teuchos::CommandLineProcessor clp(false, false);
  clp.setOption("nparts",&num_particles_per_element,"Number of particles per element");
  clp.setOption("mesh",&meshfile,"Input mesh file");
  clp.setOption("dt",&dt,"Timestep for particles");
  clp.setOption("tfinal",&t_final,"Final run time");
  clp.setOption("output",&output,"Flag to output data, default is 0, no output");
  clp.setOption("kill-particles",&part_kill,"1 if you want to kill particles when they hit a wall");
  clp.setOption("problem", &problem, "Which problem to run, two-stream or box are the currently supported problems");
  clp.setOption("density",&density,"Density of fill");
  clp.setOption("dump_mesh",&dump_mesh,"Dump mesh data for debugging, default to 0");

  Teuchos::CommandLineProcessor::EParseCommandLineReturn cmp_retval = clp.parse(argc,argv);
  if(cmp_retval==Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
     return 0;

  { // call the destructor prior to kokkos finalize
    DataWarehouse data;
    bool periodic = false;
    if ( problem == "two-stream")
      periodic=true;
    data.create_mesh(meshfile, comm, periodic);
    data.create_particle_list(data.mesh().num_elems*num_particles_per_element*3+1000);
    data.create_ES();
    data.set_kill_particles_on_wall_impact(part_kill == 1);
    Kokkos::Impl::Timer timer;

    if (problem == "two-stream"){
      VECTOR v(3,0);
      v[0] = 0.5;
      FLOAT temp=-.05;
      ParticleFill fill1a(data, num_particles_per_element/2, 0, density/2, v,temp);
      v[0] *= -1;
      ParticleFill fill1b(data, num_particles_per_element/2, 0, density/2, v,temp);
      int nparts = data.particles_reference().num_particles();
      ParticleFill fill2(data, num_particles_per_element,    2, density, VECTOR(3,0), 0);
    } else {
      ParticleFill fill1(data, num_particles_per_element, 0, density);
      ParticleFill fill2(data, num_particles_per_element, 1, density);
    }
    if ( dump_mesh )
      data.dump_mesh_data();
    PIC pic(data);

    double avg_time, fill_time = timer.seconds();
    GO total_parts, local_parts = data.particles().num_particles();
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM, 1, &fill_time, &avg_time);
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM, 1, &local_parts, &total_parts);
    if (comm->getRank() == 0 )
      printf("Fill time %lf for %ld particles\n",avg_time/comm->getSize(), (long)total_parts);

    Kokkos::Impl::Timer timer2;
    int output_cnt=0;
    for (double t=0; t<t_final; t += dt) {
      if (output && ( output_cnt ++)%output ==0 ){
        char* outBuffer = (char*) malloc(sizeof(char) * 256);
        sprintf(outBuffer, "output.%f", t);
        string output(outBuffer);
        free(outBuffer);
        ofstream out(output, ios::out);
        int nparts = data.particles_reference().num_particles();
        for (unsigned i=0; i< nparts; i+=num_particles_per_element)
          if (data.particles_reference().type(i) == 0)
            out << data.particles_reference().x(i)[0]<<" "<<data.particles_reference().x(i)[1]<<" "<<data.particles_reference().x(i)[2]<<" "<<
            data.particles_reference().v(i)[0]<<" "<<data.particles_reference().v(i)[1]<<" "<<data.particles_reference().v(i)[2]<<endl;
      }
      pic.time_step(dt);
    }

    fill_time = timer2.seconds();
    local_parts = data.particles().num_particles();
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM, 1, &local_parts, &total_parts);
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_SUM, 1, &fill_time, &avg_time);
    fill_time = avg_time / comm->getSize();
    if (comm->getRank() == 0 )
      printf("Move time %lf for %d parts, or %5.1lfE6 updates/second \n",fill_time, total_parts, 1e-6*data.mesh().num_global_elems*num_particles_per_element*t_final/(dt*fill_time) );

    pic.output_timers();
  }
  Kokkos::finalize();
  return 0;
}

