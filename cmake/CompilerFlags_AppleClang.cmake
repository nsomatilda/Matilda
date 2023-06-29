###########################
# Set target architecture #
###########################

if( MARCH )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH}")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=broadwell")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=skylake-avx512")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=cascadelake")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver2")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver4")
endif()

#########################################
# Enable specific optimization features #
#########################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -fopenmp-simd")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fveclib=AMDLIBM")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fveclib=libmvec")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fveclib=SVML")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass=loop-vectorize")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Rpass-missed=loop-vectorize")

############################
# Enable specific warnings #
############################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat-pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-documentation")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-double-promotion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")

####################
# Enable sanitizer #
####################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fno-omit-frame-pointer")

######################
# Enable GDB symbols #
######################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
