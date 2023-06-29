if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  include(cmake/CompilerFlags_Gnu.cmake)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  include(cmake/CompilerFlags_AppleClang.cmake)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  include(cmake/CompilerFlags_Clang.cmake)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
  include(cmake/CompilerFlags_IntelLLVM.cmake)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  include(cmake/CompilerFlags_Intel.cmake)
endif()



