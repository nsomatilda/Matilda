###########################
# Set target architecture #
###########################

if( MARCH )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xAVX")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xBROADWELL")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCore-AVX512 -qopt-zmm-usage=high" )
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xMIC-AVX512 -qopt-zmm-usage=high" )
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xSANDYBRIDGE" )
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xSKYLAKE-AVX512 -qopt-zmm-usage=high")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xICELAKE-SERVER")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 -fma")
endif()

#########################################
# Enable specific optimization features #
#########################################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-zmm-usage=high") # only with AVX-512, may be not beneficial.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model fast=2")     # This is enabled in -Ofast and -fast
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")                 # This is enabled in -fast
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")              # This is enabled in -fast
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prof-gen")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prof-use")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=cg,ipo,loop,vec")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5 -qopt-report-phase=vec")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=5")

############################
# Enable specific warnings #
############################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10441") # Don't warn about deprecation of classic ICC



