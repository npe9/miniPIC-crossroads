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
#ifndef _TYPES_HPP_
#define _TYPES_HPP_

#include <vector>
#include "const_len_vector.h"
#include "KokkosCompat_ClassicNodeAPI_Wrapper.hpp"
#include "Kokkos_View.hpp"
#include "Teuchos_Comm.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_CrsGraph.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_Import.hpp"

#define MAX_TEAM_SIZE 256

typedef int LO;
typedef long GO;
typedef LO P_LO;
typedef GO P_GO;
typedef double FLOAT;
typedef std::vector<GO> GO_STD_VECTOR;
typedef std::vector<LO> LO_STD_VECTOR;
typedef std::vector<FLOAT> FLOAT_1D_STD_VECTOR;

typedef ConstLenVector<FLOAT, 3> VECTOR;
typedef ConstLenVector<LO, 3> LO_VECTOR;
typedef ConstLenVector<GO, 3> GO_VECTOR;

typedef Kokkos::View<FLOAT*   > FLOAT_1D_ARRAY;
typedef Kokkos::View<FLOAT**  > FLOAT_2D_ARRAY;
typedef Kokkos::View<FLOAT*** > FLOAT_3D_ARRAY;

typedef Kokkos::View<VECTOR*   > FLOAT_1D_VEC_ARRAY;
typedef Kokkos::View<VECTOR**  > FLOAT_2D_VEC_ARRAY;
typedef Kokkos::View<VECTOR*** > FLOAT_3D_VEC_ARRAY;

typedef Kokkos::View<GO*> GO_ARRAY;
typedef Kokkos::View<GO**> TEMP_CONNECTIVITY_ARRAY;
typedef Kokkos::View<GO**> CONNECTIVITY_ARRAY;


typedef Kokkos::View<const FLOAT**[3][3], Kokkos::MemoryTraits<Kokkos::RandomAccess> > JACOBIAN_ARRAY;
typedef Kokkos::View<FLOAT**[3][3] > TEMP_JACOBIAN_ARRAY;

typedef Kokkos::DefaultExecutionSpace DeviceSpace;
typedef Kokkos::Serial HostSpace;
typedef Kokkos::Compat::KokkosDeviceWrapperNode<DeviceSpace> NodeType;
typedef Kokkos::Compat::KokkosDeviceWrapperNode<HostSpace> HostNodeType;
typedef Kokkos::TeamPolicy<DeviceSpace> TeamPolicy;
typedef TeamPolicy::member_type member_type;


typedef Tpetra::Map<LO, GO, HostNodeType> Map;
typedef Tpetra::Map<LO, GO, NodeType> DeviceMap;

typedef Tpetra::CrsGraph<LO, GO, HostNodeType> Graph;
typedef Tpetra::Vector<FLOAT, LO, GO, HostNodeType> Vector;
typedef Tpetra::Vector<FLOAT, LO, GO, NodeType> DeviceVector;
typedef Tpetra::Vector<FLOAT, LO, GO, NodeType>::dual_view_type VectorDualVeiw;
typedef Tpetra::Vector<FLOAT, LO, GO, HostNodeType>::dual_view_type HostVectorDualView;
typedef Tpetra::CrsMatrix<FLOAT, LO, GO, HostNodeType> Matrix;
typedef Tpetra::Export<LO, GO, HostNodeType> Export;
typedef Tpetra::Import<LO, GO, HostNodeType> Import;




#endif
