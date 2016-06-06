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
 * CG_solve.hpp
 *
 *  Created on: Oct 31, 2014
 *      Author: mbetten
 */

#ifndef SRC_CG_SOLVE_HPP_
#define SRC_CG_SOLVE_HPP_

#include "types.hpp"
#include "Teuchos_RCP.hpp"
#include <cmath>
#include <MatrixMarket_Tpetra.hpp>



/// Simple Conjugate gradient solver
/// @param m Operator for the system
/// @param rhs Right hand side vector
/// @param x Solution vector, filled with an initial guess
/// @param tol Desired tolerance for the solve
bool CG_solve(Teuchos::RCP<Matrix> m,Teuchos::RCP<Vector> rhs, Teuchos::RCP<Vector> x,FLOAT tol=1e-8)
{
  FLOAT rho0=0.0,rho=0.0, rho_old=0.0, beta=0.0, alpha=0.0;

  rho0 = rhs->dot(*rhs);

  Vector p(rhs->getMap()), q(rhs->getMap()), r(rhs->getMap());
  int rank = m->getComm()->getRank();
  m->apply(*x, r);
  r.update(1.0, *rhs, -1.0);

  // rho = r.r
  rho = r.dot(r);

  if ( rank == 0 )
    std::cout << 0 << " 1.0" << std::endl;

  if (rho == 0 )
    return 0;
  int iter=0, max_iter =1000;
  for (;iter<max_iter; ++iter ) {


    if ( iter == 0 )
      p.assign(r);
    else {
      beta = rho/rho_old;
      p.update(1.0, r, beta); // p <- beta*p+r
    }

    // q = Ap
    m->apply(p, q);

    alpha = p.dot(q);
    alpha = rho / alpha;//p.q


    // x+= alpha*p
    x->update(alpha, p, 1.0);


    //r -= alpha*q;
    r.update(-alpha, q, 1.0);

    rho_old = rho;
    // rho = r.r
    rho = r.dot(r);


    if(rho/rho0<tol*tol) {
      if ( rank == 0 )
        std::cout << iter << " " << sqrt(rho/rho0) <<" "<<alpha << " "<<rho<<" "<<beta<<"\n";
      break;
    }

    if ( (iter+1)%10 == 0 )
      if ( rank == 0 )
        std::cout << iter+1 << " " << sqrt(rho/rho0) <<" "<<alpha << " "<<rho<<" "<<beta<<"\n";
  }
  return iter < max_iter;
}
int BiCGStab_solve(Teuchos::RCP<Matrix> m,Teuchos::RCP<Vector> rhs, Teuchos::RCP<Vector> x,FLOAT tol=1e-8)
{

  FLOAT rho0=0.0,rho=0.0, rho_old=0.0, beta=0.0, alpha=0.0, omega=0.0, resid=0.0;

  rho0 = rhs->dot(*rhs);

#if 0
  Tpetra::MatrixMarket::Writer<Matrix>::writeSparseFile ("matrix.mm", m);
  Tpetra::MatrixMarket::Writer<Vector>::writeDenseFile ("r.mm", rhs);
#endif
  int rank = m->getComm()->getRank();

  Vector r(rhs->getMap()),r_tilde(rhs->getMap()),  p(rhs->getMap()), q(rhs->getMap()), s(rhs->getMap()), v(rhs->getMap()), t(rhs->getMap()), dummy(rhs->getMap());

  x->scale(0.);
  m->apply(*x, r);
  // Update(alpha, A, beta) is  this <- beta*this+alpha*A
  r.update(1.0, *rhs, -1.0);  // r = rhs - A*x
  r_tilde.assign(r);
  // rho = r.r
  rho = r_tilde.dot(r);

  if ( rank == 0 )
    std::cout << 0 << " 1.0" << std::endl;

  if (rho == 0 )
    return 0;

  int iter=0, max_iter = 1000;
  for (;iter<max_iter; ++iter ) {


    if ( iter == 0 )
      p.assign(r);
    else {
      rho = r_tilde.dot(r);
      if ( rho == 0 ) {
        std::cout << "Premature exit with "<<sqrt(resid/rho0) <<" error\n";
        break;
      }
      beta = rho/rho_old*alpha/omega;
      p.update(-omega, v, 1.0); //p = p-omega*v
      p.update(1.0, r, beta);    //p = r+beta*p = r + beta*(p-omega*v)
    }

    // v = Ap
    m->apply(p, v);

    //alpha = rho/(v.r-tilde)
    alpha = rhs->dot(v);
    alpha = rho / alpha;

    r.update(-alpha, v, 1.0);

    if ( (resid = r.dot(r)) < rho0*tol*tol) {
      x->update(alpha, p, 1.0);
      if ( rank == 0 )
        std::cout << iter << " " << sqrt(resid/rho0) <<" "<<alpha << " "<<rho<<" "<<beta<<"\n";
      break;
    }
    // t = A*r
    m->apply(r, t);

    //omega = t.r/t.t
    FLOAT tdott = t.dot(t);
    omega = t.dot(r)/tdott;
    if ( omega == 0.0 || tdott == 0 ) {
      std::cout <<"Premature exit\n";
      break;
    }
    // x+= alpha*p+omega*r
    x->update(alpha, p, 1.0);
    x->update(omega, r, 1.0);

    //r = r - omega*t
    r.update(-omega, t, 1.0);

    rho_old = rho;
    resid = r.dot(r);

    if(resid/rho0<tol*tol) {
      if ( rank == 0 )
        std::cout << iter << " " << sqrt(resid/rho0) <<" "<<alpha << " "<<rho<<" "<<beta<<"\n";
      break;
    }

    if ( (iter+1)%50 == 0 )
      if ( rank == 0 )
        std::cout << iter+1 << " " << sqrt(resid/rho0) <<" "<<alpha << " "<<rho<<" "<<beta<<" " << omega<<"\n";
  }
}

#endif /* SRC_CG_SOLVE_HPP_ */
