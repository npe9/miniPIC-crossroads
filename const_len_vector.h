#ifndef CONST_LEN_VECTOR_H_
#define CONST_LEN_VECTOR_H_
#include <string.h>
#include <cassert>
#include "Kokkos_Core.hpp"
  /// this class is a vector with constant length and some error checking
  /// It is useful for replacing legacy vectors and has lower storage

template < typename T, unsigned int vector_len >
class ConstLenVector {
protected:

  T data_[vector_len];
public:
  KOKKOS_INLINE_FUNCTION
  ConstLenVector() {}

  /// Constructor with size i,
  /// @param i input size of vector, has to be the same as template parameter vector_len
  KOKKOS_INLINE_FUNCTION
  ConstLenVector(int i) {resize(i);}

  /// Constructor with size i,
  /// @param i input size of vector, has to be the same as template parameter vector_len
  /// @param val input initial value for vector
  KOKKOS_INLINE_FUNCTION
  ConstLenVector(int i,const T &val) {resize(i,val);}

  /// Copy constructor
  KOKKOS_INLINE_FUNCTION
  ConstLenVector(const ConstLenVector<T,vector_len> &src) {
    for (unsigned i=0;i<vector_len; ++i)
      data_[i] = src.data_[i];
  }

  /// Assignment operator
  KOKKOS_INLINE_FUNCTION
  ConstLenVector<T,vector_len> & operator = (const ConstLenVector<T,vector_len> &src) {
    for (unsigned i=0;i<vector_len; ++i)
      data_[i]  = src.data_[i] ;
    return *this;
  }

  /// Assignment operator with a std::vector
  KOKKOS_INLINE_FUNCTION
  ConstLenVector<T,vector_len> & operator = (const std::vector<T> &src) {
    assert(src.size() <= vector_len);
    for (unsigned i=0;i<vector_len; ++i)
      data_[i]  = src[i] ;
    return *this;
  }

  /// Output the size of the vector
  KOKKOS_INLINE_FUNCTION
  unsigned int size() const {return vector_len;}

  /// Access operator
  KOKKOS_INLINE_FUNCTION
  T& operator [] (int i) { 
    assert (i>=0 && (unsigned)i < vector_len );
    return data_[i];
  }

  /// Access operator
  KOKKOS_INLINE_FUNCTION
  T& operator [] (unsigned i) { 
    assert (i < vector_len );
    return data_[i];
  }
  
  /// Access operator
  KOKKOS_INLINE_FUNCTION
  T& operator [] (unsigned long i) {
    assert (i < (unsigned long)vector_len );
    return data_[i];
  }

  /// Access operator
  KOKKOS_INLINE_FUNCTION
  const T& operator [] (int i)const { 
    assert (i>=0 && (unsigned)i < vector_len );
    return data_[i];
  }

  /// Access operator
  KOKKOS_INLINE_FUNCTION
  const T& operator [] (unsigned i)const { 
    assert (i < vector_len );
    return data_[i];
  }

  /// Access operator
 KOKKOS_INLINE_FUNCTION
  const T& operator [] (unsigned long i)const {
     assert (i < (unsigned long)vector_len );
     return data_[i];
   }

   /// cast operator so one can pass this vector as a point just with foo, not &foo[0]
  KOKKOS_INLINE_FUNCTION
  operator T*() {
    return data_;
  }

  /// cast operator so one can pass this vector as a point just with foo, not &foo[0]
  KOKKOS_INLINE_FUNCTION
  operator const T*() const {
    return data_;
  }

  /// THis is a non-op but needs to be here to conform to std::vector
  KOKKOS_INLINE_FUNCTION
  void resize(unsigned new_size) {
    assert (new_size <=vector_len);
  }

  /// THis is a constant assignment operator
 KOKKOS_INLINE_FUNCTION
  void resize(unsigned new_size, const T& val) {
    assert (new_size <=vector_len);
    for (unsigned i=0;i<new_size; ++i)
      data_[i] = val;
  }
};

 typedef ConstLenVector<double, 3> vector_double3;
 typedef ConstLenVector<double, 4> vector_double4;
 typedef ConstLenVector<double, 5> vector_double5;
 typedef ConstLenVector<double, 9> vector_double9;
 typedef ConstLenVector<int, 3> vector_int3;
 typedef ConstLenVector<int, 4> vector_int4;
 typedef ConstLenVector<ConstLenVector<double, 3>, 4> vector_double4_3;
 typedef ConstLenVector<ConstLenVector<double, 2>, 3> vector_double3_2;
 typedef ConstLenVector<std::vector<double>, 4 > vector_doublep4;
 typedef ConstLenVector<std::vector<int>, 4 > vector_intp4;
 typedef ConstLenVector<std::vector<double>, 3 > vector_doublep3;
 typedef ConstLenVector<std::vector<int>, 3 > vector_intp3;




#endif
