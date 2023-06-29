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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd" )

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-missed")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-note")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-vec-optimized")

############################
# Enable specific warnings #
############################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdisabled-optimization")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdouble-promotion")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winline")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpadded")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstack-protector")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-attribute=pure")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-attribute=const")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-attribute=noreturn")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-attribute=malloc")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wunreachable-code")

####################
# Enable sanitizer #
####################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-omit-frame-pointer")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")


#####################
# Make GCC pedantic #
#####################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic")

###########################
# Create assembler output #
###########################

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -fverbose-asm" )
