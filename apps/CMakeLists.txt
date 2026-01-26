
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR})

add_executable(matilda_test matilda_test.cc)
target_link_libraries(matilda_test matilda ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})



set(BLA_STATIC OFF)

set(BLA_VENDOR Intel10_64lp) # multi-threaded
find_package(BLAS)
if( BLAS_FOUND )
  add_executable(matilda_test.mkl matilda_test.cc)
  target_compile_definitions( matilda_test.mkl PRIVATE BENCHMARKING_BLAS=1 BLAS_PROVIDER="MKL" BLAS_MKL=1 )
  target_link_libraries(matilda_test.mkl matilda ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES})
  message( STATUS "Going to build matilda_test.mkl" )
endif()


set(BLA_VENDOR OpenBLAS)
set(CMAKE_LIBRARY_PATH ${OPENBLAS_ROOT})
find_package(BLAS)
if( BLAS_FOUND )
  include_directories( ${BLAS_INCLUDES} )
  add_executable(matilda_test.openblas matilda_test.cc)
  target_compile_definitions( matilda_test.openblas PRIVATE BENCHMARKING_BLAS=1 BLAS_PROVIDER="OpenBLAS" BLAS_OPENBLAS=1 )
  target_link_libraries(matilda_test.openblas matilda ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES})
  message( STATUS "Going to build matilda_test.openblas" )
endif()


set(BLA_VENDOR AOCL_mt)
set(CMAKE_LIBRARY_PATH ${AOCL_ROOT})
find_package(BLAS)
if( BLAS_FOUND )
  include_directories( ${BLAS_INCLUDES} )
  add_executable(matilda_test.aocl matilda_test.cc)
  target_compile_definitions( matilda_test.aocl PRIVATE BENCHMARKING_BLAS=1 BLAS_PROVIDER="AOCL" BLAS_AOCL=1 )
  target_link_libraries(matilda_test.aocl matilda ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS} ${BLAS_LIBRARIES})
  message( STATUS "Going to build matilda_test.aocl" )
endif()
