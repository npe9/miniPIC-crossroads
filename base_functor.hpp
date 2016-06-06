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
 * base_functor.hpp
 *
 *  Created on: Sep 16, 2014
 *      Author: mbetten
 */

#ifndef BASE_FUNCTOR_HPP_
#define BASE_FUNCTOR_HPP_
#include <iostream>
#include <string>
#include <vector>

#include "types.hpp"
#include "Kokkos_View.hpp"

class BaseKokkosFunctor {
public:
  BaseKokkosFunctor(std::string name) :
    name_(name), error_code_("BaseKokkosFunctor::error_code_") ,error_count_("BaseKokkosFunctor::error_count_") {
  }
  virtual ~BaseKokkosFunctor(){
  }

  virtual void report_error() {

    if (error_code_() == 0 ) return;
    if ( error_code_() >= static_cast<int>(errors_.size()) ) {
      std::cerr << "Unknown error last code "<<error_code_()<<" reported, total error count is "<<error_count_()<< "\n";
      return;
    }
    std::cerr << "Last error code "<<error_code_()<<" \""<<errors_[error_code_()]<< "\" reported, total error count is "<<error_count_()<< "\n";

  }

  KOKKOS_INLINE_FUNCTION
  void push_error( int error_code) const {
    error_code_() = error_code;
    error_count_() ++;
  }


protected:
  //static TeamPolicy policy;
  std::string name_;
  Kokkos::View<int> error_code_;
  Kokkos::View<int> error_count_;
  std::vector<std::string> errors_;
};



#endif /* BASE_FUNCTOR_HPP_ */
